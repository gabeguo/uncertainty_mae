from models_mae import MaskedAutoencoderViT, mae_vit_base_patch16_dec512d8b
import copy
import torch
import torch.nn as nn

class MultiHeadViT(nn.Module):
    """
    Creates ViT Encoder from pre-trained MAE backbone.
    Backbone is shared (and weights are backprop-able), 
    but last few layers are copied and unfrozen,
    to use as the three heads.
    """
    def __init__(self, backbone_path, num_unshared_layers=1, freeze_backbone=False, return_all_tokens=False):
        super().__init__()

        self.backbone_path = backbone_path

        backbone = mae_vit_base_patch16_dec512d8b()
        backbone.load_state_dict(torch.load(backbone_path)['model'])

        assert isinstance(backbone, MaskedAutoencoderViT)

        self.backbone = backbone
        self.num_unshared_layers = num_unshared_layers
        self.freeze_backbone = freeze_backbone
        self.return_all_tokens = return_all_tokens

        self.create_layers()

        return
    
    def create_layers(self):
        # Make divergent heads
        self.low_head = self.create_head()
        self.mid_head = self.create_head()
        self.high_head = self.create_head()

        # (Optionally) freeze backbone AFTER copying weights
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print('froze backbone')

        return

    def create_head(self):
        """
        Copies the last [self.num_unshared_layers] blocks + the norm layer of the backbone,
        and makes learnable/unshared head.
        """
        print('creating head')
        # Copy from backbone
        copied_blocks = copy.deepcopy(self.backbone.blocks)
        assert self.num_unshared_layers <= len(copied_blocks)      
        unfrozen_blocks = copied_blocks[-self.num_unshared_layers:]
        # norm (with learnable bias and scale)
        norm = copy.deepcopy(self.backbone.norm)

        head = nn.Sequential(*unfrozen_blocks, norm)

        return head


    def forward(self, x):
        """
        Mostly lifted from models_mae forward_encoder, except:
        (1) Returns lower bound, point estimate, upper bound
        (2) Have option to return only cls token
        """
        B = x.shape[0]
        # embed patches
        x = self.backbone.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.backbone.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.backbone.random_masking(x, mask_ratio=0)

        # append cls token
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply (shared) Transformer blocks
        for blk in self.backbone.blocks[:-self.num_unshared_layers]:
            x = blk(x)
        # apply (learnable) head blocks
        x_low = self.low_head(x)
        x_mid = self.mid_head(x)
        x_high = self.high_head(x)
        
        the_outputs = [x_low, x_mid, x_high]

        for idx in range(len(the_outputs)):
            curr_x = the_outputs[idx]
            assert curr_x.shape == (B, 14*14+1, 768) or curr_x.shape == (B, 16*16+1, 768)

            if not self.return_all_tokens:
                curr_x = curr_x[:, 0] # cls token
                assert curr_x.shape == (B, 768)
                the_outputs[idx] = curr_x
        
        return the_outputs[0], the_outputs[1], the_outputs[2]

class EncoderViT(nn.Module):
    """
    Takes the encoder from the MAE.
    Can initialize to pre-trained weights, if you like.
    Can also freeze, if you like.
    """
    def __init__(self, backbone_path=None, freeze_backbone=False, return_all_tokens=False):
        super().__init__()

        self.backbone_path = backbone_path
        self.freeze_backbone = freeze_backbone

        # create backbone
        backbone = mae_vit_base_patch16_dec512d8b()
        # optionally load backbone weights
        if backbone_path is not None:
            the_state_dict = torch.load(backbone_path)
            if 'model' in the_state_dict:
                the_state_dict = the_state_dict['model']
            backbone.load_state_dict(the_state_dict)
        assert isinstance(backbone, MaskedAutoencoderViT)
        self.backbone = backbone

        # freeze backbone
        if freeze_backbone:
            backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # whether to return all tokens, or just the cls token.
        self.return_all_tokens = return_all_tokens
        return
    
    def forward(self, x, mask_ratio=0):
        B = x.shape[0]
        x, _, _ = self.backbone.forward_encoder(x, mask_ratio=mask_ratio)
        assert x.shape == (B, 14*14+1, 768) or x.shape == (B, 16*16+1, 768)
        if not self.return_all_tokens:
            x = x[:,0] # cls token
            assert x.shape == (B, 768)
        return x
    
    def get_mask_noise(self, x):
        """
        Gets fixed noise that can be used across models
        """
        with torch.no_grad():
            x = self.backbone.patch_embed(x)
            N, L, D = x.shape  # batch, length, dim
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        return noise

    def forward_fixed_mask(self, x, mask_ratio, noise):
        """
        Lifted from models_mae.py, but masking is controllable
        """
        # embed patches
        x = self.backbone.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.backbone.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, noise)

        # append cls token
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        return x, mask, ids_restore
    
    def random_masking(self, x, mask_ratio, noise):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
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

if __name__ == "__main__":
    backbone_path = '/home/gabeguo/vae_mae/cifar100_train/checkpoint-399.pth'

    # dummy data
    x = torch.rand(4, 3, 224, 224)

    # create teacher
    teacher = TeacherViT(backbone_path=backbone_path, return_all_tokens=False)
    print(teacher(x).shape)
    teacher_learnable_params = {item[0] for item in teacher.named_parameters() if item[1].requires_grad}
    print('learnable teacher params:', len(teacher_learnable_params))

    # create student
    model = MultiHeadViT(backbone_path=backbone_path, 
                         num_unshared_layers=1, freeze_backbone=False, return_all_tokens=False)

    print(model)

    print([item.shape for item in model(x)])

    learnable_parameters = {x[0] for x in model.named_parameters() if x[1].requires_grad}
    all_parameters = {x[0] for x in model.named_parameters()}

    assert len([y1 for y1 in model.named_parameters()]) == len([y2 for y2 in model.parameters()])

    non_learnable_parameters = all_parameters - learnable_parameters
    assert all(['backbone.' in the_param for the_param in non_learnable_parameters])
    # print('non-learnable parameters:', non_learnable_parameters)
    # print('learnable parameters:', learnable_parameters)
    
    print('num learnable parameters:', len(learnable_parameters))
    print('num parameters:', len(all_parameters))
