import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT

class UncertaintyMAE(nn.Module):
    def __init__(self, visible_mae, invisible_mae):
        super().__init__()

        assert isinstance(visible_mae, MaskedAutoencoderViT)
        assert isinstance(invisible_mae, MaskedAutoencoderViT)
        assert invisible_mae.vae # should be variational
        assert not visible_mae.vae # should not be variational

        self.visible_mae = visible_mae
        self.invisible_mae = invisible_mae

        return
    
    def forward(self, imgs, mask_ratio=0.75, force_mask=None):
        """
        Returns:
        loss, pred, mask
        """
        # TODO: eval mode

        N, L, D = imgs.shape
        len_keep = int(L * (1 - mask_ratio))
        len_remove = L - len_keep

        # Force Mask
        mask_layout = torch.ones(14, 14).to(device=imgs.device)
        mask_layout = mask_layout.flatten()
        mask_layout[torch.multinomial(mask_layout, len_remove, replacement=False)] = 0
        keep_indices = torch.where(mask_layout == 1)[0]
        mask_indices = torch.where(mask_layout == 0)[0]
        keep_indices = keep_indices.reshape(1, -1).expand(N, -1)
        mask_indices = mask_indices.reshape(1, -1).expand(N, -1)
        
        ids_shuffle = torch.cat((keep_indices, mask_indices), dim=1)
        visible_latent, mask, ids_restore = self.visible_mae.forward_encoder(imgs, mask_ratio, 
                                                                             force_mask=ids_shuffle)
        # call invisible encoder
        ids_reverse_shuffle = torch.cat((mask_indices, keep_indices), dim=1)
        invisible_latent, reverse_mask, reverse_ids_restore = self.invisible_mae.forward_encoder(imgs, 1 - mask_ratio, 
                                                                                                 force_mask=ids_reverse_shuffle)
        assert invisible_latent.shape[1] + visible_latent.shape[1] == imgs.shape[1] + 2, \
            f"invisible_latent: {invisible_latent.shape}, visible latent: {visible_latent.shape}, imgs: {imgs.shape}"
        assert invisible_latent.shape[0] == visible_latent.shape[0]
        assert invisible_latent.shape[2] == visible_latent.shape[2]
        assert torch.sum(reverse_mask) + torch.sum(mask) == 14 * 14
        
        # TODO: if this gets buggy, try to regenerate the real image with these indices

        pred = self.visible_mae.forward_decoder(visible_latent, ids_restore, force_mask_token=invisible_latent)  # [N, L, p*p*3]
        assert pred.shape == (N, L, 3), f"pred.shape is {pred.shape}"
        loss = self.visible_mae.forward_loss(imgs, pred, mask)

        return loss, pred, mask
