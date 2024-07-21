import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT
import random
import numpy as np

class UncertaintyMAE(nn.Module):
    def __init__(self, visible_mae, invisible_mae, dropout_ratio=0, 
                 load_weights=None, same_encoder=False, end_to_end_finetune=False,
                 block_mask_prob=0.5, var=1):
        super().__init__()

        self.same_encoder = same_encoder
        self.load_weights = load_weights
        self.end_to_end_finetune = end_to_end_finetune
        self.block_mask_prob = block_mask_prob
        self.var = var
        print(f'Same encoder: {self.same_encoder}')
        print(f'End-to-end finetune: {self.end_to_end_finetune}')
        print(f'Block mask prob: {self.block_mask_prob}')

        assert isinstance(invisible_mae, MaskedAutoencoderViT)
        assert invisible_mae.vae

        if self.same_encoder:
            assert visible_mae is None

            self.visible_mae = invisible_mae
            self.invisible_mae = invisible_mae

        else:
            assert isinstance(visible_mae, MaskedAutoencoderViT)
            assert not visible_mae.vae # should be deterministic

            self.visible_mae = visible_mae
            self.invisible_mae = invisible_mae

        if self.load_weights is not None:
            if self.end_to_end_finetune:
                checkpoint = torch.load(load_weights, map_location='cpu')
                msg = self.load_state_dict(checkpoint['model'], strict=True)
                print(f'CONTINUING TO FINETUNE FROM: {load_weights}')
            else:
                self.visible_mae.adopt_weights(load_weights)
                self.invisible_mae.adopt_weights(load_weights)

        self.dropout_ratio = dropout_ratio

        return
    
    def block_mask(self, mask_layout):
        random_float = random.random()
        if random.random() < 0.5: # mask out 75%
            mask_ratio = 0.75 # just for this run
            if random_float < 0.25:
                mask_layout[0:14, 0:7] = 0
                mask_layout[0:7, 7:14] = 0
            elif random_float < 0.5:
                mask_layout[0:14, 0:7] = 0
                mask_layout[7:14, 7:14] = 0
            elif random_float < 0.75:
                mask_layout[0:14, 7:14] = 0
                mask_layout[0:7, 0:7] = 0
            else:
                mask_layout[0:14, 7:14] = 0
                mask_layout[7:14, 0:7] = 0
        else: # mask out 50%
            mask_ratio = 0.5
            if random_float < 0.25:
                mask_layout[0:14, 0:7] = 0
            elif random_float < 0.5:
                mask_layout[0:14, 7:14] = 0
            elif random_float < 0.75:
                mask_layout[0:7, 0:14] = 0
            else:
                mask_layout[7:14, 0:14] = 0
        return mask_ratio

    def forward(self, imgs, mask_ratio=0.75, force_mask=None, return_component_losses=False,
                add_default_mask=False, print_stats=False):
        """
        Returns:
        loss, pred, mask
        """
        # TODO: eval mode
        N = imgs.shape[0]
        L = 14 * 14
        len_keep = int(L * (1 - mask_ratio))
        len_remove = L - len_keep

        # Force Mask
        if force_mask is None:
            mask_layout = torch.ones(14, 14).to(device=imgs.device)
            if random.random() < self.block_mask_prob:
                mask_ratio = self.block_mask(mask_layout) # need to keep track of alt mask ratio
                mask_layout = mask_layout.flatten()
            else:
                mask_layout = mask_layout.flatten()
                mask_layout[torch.multinomial(mask_layout, len_remove, replacement=False)] = 0
            
            keep_indices = torch.where(mask_layout == 1)[0]
            mask_indices = torch.where(mask_layout == 0)[0]
            # does it by batch
            keep_indices = keep_indices.reshape(1, -1).expand(N, -1)
            mask_indices = mask_indices.reshape(1, -1).expand(N, -1)
        else:
            keep_indices, mask_indices = force_mask
        
        ids_shuffle = torch.cat((keep_indices, mask_indices), dim=1)
        if self.same_encoder:
            assert self.visible_mae.vae
            _, mask, ids_restore, visible_latent, _ = \
                self.visible_mae.forward_encoder(imgs, mask_ratio, force_mask=ids_shuffle)
        else:
            visible_latent, mask, ids_restore = \
                self.visible_mae.forward_encoder(imgs, mask_ratio, force_mask=ids_shuffle)

        # use invisible encoder
        if self.training and random.random() > self.dropout_ratio: 
            ids_reverse_shuffle = torch.cat((mask_indices, keep_indices), dim=1)
            reverse_mask_ratio = ( (1 - mask_ratio) * L - 0.5 ) / L
            invisible_latent, reverse_mask, _, latent_mean, latent_log_var = \
                self.invisible_mae.forward_encoder(imgs, reverse_mask_ratio, force_mask=ids_reverse_shuffle)
            assert invisible_latent.shape[1] + visible_latent.shape[1] == 14 * 14 + 2, \
                f"invisible_latent: {invisible_latent.shape}, visible latent: {visible_latent.shape}, imgs: {imgs.shape}"
            assert invisible_latent.shape[0] == visible_latent.shape[0]
            assert invisible_latent.shape[2] == visible_latent.shape[2]
            assert torch.sum(reverse_mask) + torch.sum(mask) == N * 14 * 14, f"reverse mask: {torch.sum(reverse_mask)}, {torch.sum(mask)}"
            # kld_loss = -0.5 * \
            #     torch.mean(1 + latent_log_var - latent_mean.pow(2) - torch.minimum(latent_log_var.exp(), torch.full_like(latent_log_var, 100)))
            # Sanity checked this one with commented code above
            kld_loss = 0.5 * \
                torch.mean(
                    np.log(self.var) - latent_log_var - 1 + \
                    1 / self.var * (latent_mean.pow(2) + latent_log_var.exp())
                )
            if random.random() < 0.01:
                print('mean:', latent_mean.mean())
                print('var:', latent_log_var.exp().mean())
        # test time
        else:
            invisible_num_tokens = 14 * 14 + 2 - visible_latent.shape[1]
            # invisible_latent = torch.randn(visible_latent.shape[0], invisible_num_tokens, visible_latent.shape[2],
            #                                device=visible_latent.device)
            invisible_latent = torch.normal(mean=0, std=self.var**0.5, 
                            size=(visible_latent.shape[0], invisible_num_tokens, visible_latent.shape[2])
            )
            invisible_latent = invisible_latent.to(visible_latent.device)
            kld_loss = 0
        
        if print_stats:
            print(f'pre-embedding invisible latent: mean = {invisible_latent.mean():.3f}; std = {invisible_latent.std():.3f}')
            if self.training:
                print(f'logvar: {latent_log_var.mean():.3f}')
                print(f'mean: {latent_mean.mean():.3f}')

        # TODO: if this gets buggy, try to regenerate the real image with these indices
        invisible_latent = self.invisible_mae.decoder_embed(invisible_latent) # embed for decoder
        pred = self.visible_mae.forward_decoder(visible_latent, ids_restore, 
                                                force_mask_token=invisible_latent,
                                                add_default_mask=add_default_mask,
                                                print_stats=print_stats)  # [N, L, p*p*3]
        assert pred.shape == (N, L, 16 * 16 * 3), f"pred.shape is {pred.shape}"
        reconstruction_loss = self.visible_mae.forward_loss(imgs, pred, mask)

        if self.training:
            loss = reconstruction_loss + self.invisible_mae.kld_beta * kld_loss
        else:
            loss = reconstruction_loss

        if return_component_losses:
            return loss, pred, mask, reconstruction_loss, kld_loss

        return loss, pred, mask
