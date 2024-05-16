import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT
import random

class UncertaintyMAE(nn.Module):
    def __init__(self, visible_mae, invisible_mae_mean, invisible_mae_log_var, kld_beta):
        super().__init__()

        assert isinstance(visible_mae, MaskedAutoencoderViT)
        assert isinstance(invisible_mae_mean, MaskedAutoencoderViT)
        assert isinstance(invisible_mae_log_var, MaskedAutoencoderViT)
        # assert invisible_mae.vae # should be variational
        assert not invisible_mae_mean.vae
        assert not invisible_mae_log_var.vae
        assert not visible_mae.vae # should not be variational

        self.visible_mae = visible_mae
        self.invisible_mae_mean = invisible_mae_mean
        self.invisible_mae_log_var = invisible_mae_log_var
        self.kld_beta = kld_beta

        return
    
    def forward(self, imgs, mask_ratio=0.75, force_mask=None):
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
            mask_layout = mask_layout.flatten()
            mask_layout[torch.multinomial(mask_layout, len_remove, replacement=False)] = 0
            keep_indices = torch.where(mask_layout == 1)[0]
            mask_indices = torch.where(mask_layout == 0)[0]
            keep_indices = keep_indices.reshape(1, -1).expand(N, -1)
            mask_indices = mask_indices.reshape(1, -1).expand(N, -1)
        else:
            keep_indices, mask_indices = force_mask
        
        ids_shuffle = torch.cat((keep_indices, mask_indices), dim=1)
        visible_latent, mask, ids_restore = self.visible_mae.forward_encoder(imgs, mask_ratio, force_mask=ids_shuffle)

        # use invisible encoder
        if self.training: 
            ids_reverse_shuffle = torch.cat((mask_indices, keep_indices), dim=1)
            invisible_latent_mean, reverse_mask, reverse_ids_restore = \
                self.invisible_mae_mean.forward_encoder(imgs, 1 - mask_ratio, force_mask=ids_reverse_shuffle)
            invisible_latent_log_var, reverse_mask, reverse_ids_restore = \
                self.invisible_mae_log_var.forward_encoder(imgs, 1 - mask_ratio, force_mask=ids_reverse_shuffle)
            # print('mean:', torch.mean(latent_mean), torch.std(latent_mean))
            # print('std:', torch.mean(latent_log_var.exp()), torch.std(latent_log_var.exp()))
            assert invisible_latent_mean.shape[1] + visible_latent.shape[1] == 14 * 14 + 2, \
                f"invisible_latent: {invisible_latent.shape}, visible latent: {visible_latent.shape}, imgs: {imgs.shape}"
            assert invisible_latent_mean.shape[0] == visible_latent.shape[0]
            assert invisible_latent_mean.shape[2] == visible_latent.shape[2]
            assert invisible_latent_mean.shape == invisible_latent_log_var.shape
            assert torch.sum(reverse_mask) + torch.sum(mask) == N * 14 * 14, f"reverse mask: {torch.sum(reverse_mask)}, {torch.sum(mask)}"
            kld_loss = -0.5 * self.kld_beta * \
                torch.mean(1 + invisible_latent_log_var - invisible_latent_mean.pow(2) 
                    - torch.minimum(invisible_latent_log_var.exp(), torch.full_like(invisible_latent_log_var, 100)))
        # use random noise, to make more robust
        else:
            invisible_num_tokens = 14 * 14 + 2 - visible_latent.shape[1]
            invisible_latent = torch.randn(visible_latent.shape[0], invisible_num_tokens, visible_latent.shape[2],
                                           device=visible_latent.device)
            kld_loss = 0
        invisible_latent = self.invisible_mae_mean.reparameterization(invisible_latent_mean, invisible_latent_log_var)
        # TODO: if this gets buggy, try to regenerate the real image with these indices
        invisible_latent = self.invisible_mae_mean.decoder_embed(invisible_latent) # embed for decoder
        pred = self.visible_mae.forward_decoder(visible_latent, ids_restore, force_mask_token=invisible_latent)  # [N, L, p*p*3]
        assert pred.shape == (N, L, 16 * 16 * 3), f"pred.shape is {pred.shape}"
        loss = self.visible_mae.forward_loss(imgs, pred, mask)

        if self.training:
            loss += kld_loss

        return loss, pred, mask
