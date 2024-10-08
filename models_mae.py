# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


def quantile_loss(z_pred, z_gt, q):
    unreduced_loss = torch.maximum(z_gt - z_pred, torch.zeros_like(z_gt)) * q + \
        torch.maximum(z_pred - z_gt, torch.zeros_like(z_gt)) * (1 - q)
    reduced_loss = torch.mean(unreduced_loss)
    return reduced_loss

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 quantile=None, vae=False, kld_beta=1, num_vae_blocks=1,
                 disable_zero_conv=False):
        super().__init__()

        self.vae = vae
        self.kld_beta = kld_beta

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        if self.vae:
            if num_vae_blocks == 1:
                self.block_mean = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                self.block_log_var = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            else:
                self.block_mean = nn.Sequential(*[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                    for i in range(num_vae_blocks)
                ])
                self.block_log_var = nn.Sequential(*[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) 
                    for i in range(num_vae_blocks)
                ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.quantile = quantile

        self.initialize_weights()

        self.disable_zero_conv = disable_zero_conv
        if self.vae and (not self.disable_zero_conv):
            # add zero-convolution to log-var for numerical stability; must be done AFTER weight initialization
            self.logVar_zero_conv_weight = torch.nn.Parameter(torch.zeros(1))
            self.logVar_zero_conv_weight.requires_grad = True
            self.logVar_zero_conv_bias = torch.nn.Parameter(torch.zeros(1))
            self.logVar_zero_conv_bias.requires_grad = True
            # also add to mean, to make training more stable
            self.mean_zero_conv_weight = torch.nn.Parameter(torch.zeros(1))
            self.mean_zero_conv_weight.requires_grad = True
            self.mean_zero_conv_bias = torch.nn.Parameter(torch.zeros(1))
            self.mean_zero_conv_bias.requires_grad = True

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adopt_weights(self, weights_path, freeze=True):
        checkpoint = torch.load(weights_path, map_location='cpu')
        print(f"is vae: {self.vae}")
        if not self.vae:
            msg = self.load_state_dict(checkpoint['model'], strict=True)
            missing_keys = list()
        else:
            msg = self.load_state_dict(checkpoint['model'], strict=False)
            missing_keys = msg.missing_keys
        parameters_to_train = list()
        if freeze:
            for name, param in self.named_parameters():
                if (name in missing_keys) or ('decoder' in name):
                    param.requires_grad = True
                    parameters_to_train.append(name)
                else:
                    param.requires_grad = False
        print(msg)
        print('train:', parameters_to_train)
        assert set(missing_keys).issubset(set(parameters_to_train))
        return

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, force_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if force_mask is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        else:
            ids_shuffle = force_mask
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # Thanks https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)     
        z = mean + var * epsilon
        return z

    def forward_encoder(self, x, mask_ratio, force_mask=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, force_mask=force_mask)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.vae:
            mean_x = self.block_mean(x)
            if not self.disable_zero_conv:
                mean_x = self.mean_zero_conv_weight * mean_x + self.mean_zero_conv_bias
            log_var_x = self.block_log_var(x)
            if not self.disable_zero_conv:
                log_var_x = self.logVar_zero_conv_weight * log_var_x + self.logVar_zero_conv_bias
            x = self.reparameterization(mean=mean_x, var=torch.exp(0.5 * log_var_x))
            # x = self.norm(x)
            return x, mask, ids_restore, mean_x, log_var_x
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, force_mask_token=None, add_default_mask=False,
                        print_stats=False):
        # embed tokens
        x = self.decoder_embed(x)

        # print('x embedding shape', x.shape)
        # print('mask token shape', self.mask_token.shape)
        # print('ids restore shape', ids_restore.shape)

        # append mask tokens to sequence
        default_mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        if force_mask_token is None:
            mask_tokens = default_mask_tokens
        else:
            # we expect the encoding of the masked (peeked at) tokens from the other encoder
            mask_tokens = force_mask_token[:, 1:, :] # no cls token from masked encoding
            if add_default_mask:
                mask_tokens = mask_tokens + default_mask_tokens
        if print_stats:
            print(f"latent mean: {torch.mean(x):.3f}; std: {torch.std(x):.3f}")
            print(f"mask token mean: {torch.mean(mask_tokens):.3f}; std: {torch.std(mask_tokens):.3f}")
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, latent_mean=None, latent_log_var=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        if self.quantile is not None:
            loss = quantile_loss(z_pred=pred, z_gt=target, q=self.quantile)
        else:
            loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        if (latent_mean is not None) and (latent_log_var is not None):
            kld_loss = -0.5 * self.kld_beta * torch.mean(1 + latent_log_var - latent_mean.pow(2) - torch.minimum(latent_log_var.exp(), torch.full_like(latent_log_var, 1000)))
            loss += kld_loss
        return loss

    def forward(self, imgs, mask_ratio=0.75, force_mask=None, show_variance=False,
                print_stats=False):
        forward_retVal = self.forward_encoder(imgs, mask_ratio, force_mask=force_mask)
        if self.vae:
            latent, mask, ids_restore, latent_mean, latent_log_var = forward_retVal
            if show_variance:
                print('mean:', torch.mean(latent_mean))
                print('variance:', torch.mean(latent_log_var.exp()))
        else:
            latent, mask, ids_restore = forward_retVal
        # print('latent:', latent.shape)
        # print('mask:', mask)
        # print('mask shape:', mask.shape)
        # print('ids_restore:', ids_restore)
        # print('ids_restore shape:', ids_restore.shape)
        pred = self.forward_decoder(latent, ids_restore, print_stats=print_stats)  # [N, L, p*p*3]
        if self.vae:
            loss = self.forward_loss(imgs, pred, mask, latent_mean=latent_mean, latent_log_var=latent_log_var)
        else:
            loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
