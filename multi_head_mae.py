import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT

class MultiHeadMAE(nn.Module):
    def __init__(self, lower_mae, median_mae, upper_mae):
        assert isinstance(lower_mae, MaskedAutoencoderViT)
        assert isinstance(median_mae, MaskedAutoencoderViT)
        assert isinstance(upper_mae, MaskedAutoencoderViT)

        self.lower_mae = lower_mae
        self.median_mae = median_mae
        self.upper_mae = upper_mae

        return
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.median_mae.forward_encoder(imgs, mask_ratio)
        the_preds = list()
        the_losses = list()
        for curr_model in [self.lower_mae, self.median_mae, self.upper_mae]:
            pred = curr_model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = curr_model.forward_loss(imgs, pred, mask)
            the_preds.append(pred)
            the_losses.append(loss)
        return the_losses[0], the_preds[0], \
                the_losses[1], the_preds[1], \
                the_losses[2], the_preds[2], \
                mask