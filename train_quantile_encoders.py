import torch

# Verified in Colab
def quantile_loss(q, z_pred, z_gt):
    unreduced_loss = torch.maximum(z_gt - z_pred, torch.zeros_like(z_gt)) * q + \
        torch.maximum(z_pred - z_gt, torch.zeros_like(z_gt)) * (1 - q)
    reduced_loss = torch.mean(unreduced_loss)
    return reduced_loss

def train_quantile_regressors(args, dataloader, point_estimator):
    """
    Using shared layers, except for last few
    """
    point_estimator.eval()
    for epoch in range(args.num_epochs):
        for idx, x in enumerate(dataloader):
            with torch.no_grad():
                z = point_estimator(x)
            # 
            lower_loss = quantile_loss(args.lower_quantile, lower_encoder(x), z)
            mid_loss = quantile_loss(0.5, mid)
            upper_loss = quantile_loss(args.upper_quantile, upper_encoder(x), z)
            loss = lower_loss + upper_loss
            loss.backward()


"""
Strats:
(1) Train quantile regressors from scratch
(2) Initialize quantile regressors to point estimator weights
(3) Re-train point estimator + quantile regressors with shared layers except for last few, 
    with original point estimator as teacher
"""










def train_latent_uncertainty(args, dataloader, teacher_encoder, opt, freeze_backbone=False):
    """
    (1) Use ViT encoder (not MAE) as teacher_encoder.
    (2) Uncertainty-aware backbone is initialized from teacher_encoder.
    (3) Use three different heads that share common uncertainty-aware backbone. 
        Backprop with quantile regression with teacher's latent code (cls token) as target, jointly
        through all three heads + backbone (TODO: try freezing backbone).
    (4) Return the backbone, plus the three heads. Prob use median head's cls token as representation,
        but can experiment with other two.
    """
    assert isinstance(teacher_encoder, VisionTransformer) # NOT MAE
    # init student backbone from teacher encoder weights
    backbone = teacher_encoder.copy()
    # three different heads
    # TODO: check that name is actually head
    lower_head = backbone.head.copy()
    median_head = backbone.head.copy()
    upper_head = backbone.head.copy()
    # replace orig head with ID
    backbone.head = torch.nn.Identity()

    # ready to train
    backbone.train()
    lower_head.train()
    median_head.train()
    upper_head.train()

    # freeze models
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
    for param in teacher_encoder.parameters():
        param.requires_grad = False

    # freeze teacher encoder
    teacher_encoder = teacher_encoder.eval()
    for epoch in range(args.epochs):
        for idx, img in enumerate(dataloader):
            with torch.no_grad():
                z_gt, _, _ = teacher_encoder(img)
                z_gt = z_gt.detach()
            # use shared weights to get rep
            pre_z, _, _ = backbone(img)
            # get top, middle, and upper quantiles
            low_z = lower_head(pre_z)
            mid_z = median_head(pre_z)
            high_z = upper_head(pre_z)
            # get losses
            low_loss = quantile_loss(z_pred=low_z, z_gt=z_gt, quantile=args.lower)
            high_loss = quantile_loss(z_pred=high_z, z_gt=z_gt, quantile=args.upper)
            mid_loss = quantile_loss(z_pred=mid_z, z_gt=z_gt, quantile=0.5)
            # backprop
            total_loss = low_loss + high_loss + mid_loss
            total_loss.backward()
            # optimizer step
            opt.step()
            # clear grad
            opt.zero_grad()
    
    return {BACKBONE: backbone, LOWER: lower_head, MEDIAN: median_head, UPPER: upper_head}
    

def load_vit(args):
    vit = VisionTransformer(global_pool=False)
    pass

def load_dataloader(args):
    pass

if __name__ == "__main__":
    """
    --lr
    --min_lr
    --weight_decay
    --batch_size

    --freeze_backbone
    --lower_quantile
    --upper_quantile

    --nb_classes

    --data_path
    --teacher_path
    --output_dir

    --seed

    --num_epochs
    """

        


"""
Alt ideas: 
Latent:
(1) Post-hoc: Train three different regressors:
    disadvantage is that it's not clear which one to use for representation, and this may take too much compute.

Real space:
(1) Pixel uncertainty with three different decoders:
    advantage is that all knowledge goes into encoder, but pixel space is not semantically meaningful
"""