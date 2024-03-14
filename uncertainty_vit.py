from models_mae import MaskedAutoencoderViT, mae_vit_base_patch16_dec512d8b
import copy
import torch
import torch.nn as nn

class MultiHeadViT(nn.Module):
    """
    Creates ViT Encoder from pre-trained MAE backbone.
    Backbone is frozen, but last few layers are copied and unfrozen,
    to use as the three heads.
    """
    def __init__(self, backbone, num_unfrozen_layers=1, return_all_tokens=False):
        super().__init__()

        assert isinstance(backbone, MaskedAutoencoderViT)

        self.backbone = backbone
        self.num_unfrozen_layers = num_unfrozen_layers
        self.return_all_tokens = return_all_tokens

        self.freeze_layers()

        return
    
    def freeze_layers(self):
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Make unfrozen heads
        self.low_head = self.create_head()
        self.mid_head = self.create_head()
        self.high_head = self.create_head()

        return

    def create_head(self):
        """
        Copies the last [self.num_unfrozen_layers] blocks + the norm layer of the backbone,
        and makes unfrozen head.
        """
        print('creating head')
        # Copy from backbone
        copied_blocks = copy.deepcopy(self.backbone.blocks)      
        unfrozen_blocks = copied_blocks[-self.num_unfrozen_layers:]
        # norm (with learnable bias and scale)
        norm = copy.deepcopy(self.backbone.norm)

        # Unfreeze these layers
        for blk in unfrozen_blocks:
            for param in blk.parameters():
                param.requires_grad = True
            print('\tunfroze block')
        for the_params in [norm.weight, norm.bias]:
            for param in the_params:
                param.requires_grad = True

        head = nn.Sequential(*unfrozen_blocks, norm)

        return head


    def forward(self, x):
        """
        Mostly lifted from models_mae, except:
        (1) Returns lower bound, point estimate, upper bound
        (2) Have option to return only cls token
        """
        B = x.shape[0]
        # embed patches
        x = self.backbone.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.backbone.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.backbone.random_masking(x, 0)

        # append cls token
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply (frozen) Transformer blocks
        for blk in self.backbone.blocks[:-self.num_unfrozen_layers]:
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

if __name__ == "__main__":
    mae = mae_vit_base_patch16_dec512d8b()
    mae.load_state_dict(torch.load('/home/gabeguo/vae_mae/cifar100_train/checkpoint-399.pth')['model'])

    model = MultiHeadViT(backbone=mae, num_unfrozen_layers=1, return_all_tokens=True)

    print(model)

    x = torch.rand(4, 3, 224, 224)
    print([item.shape for item in model(x)])