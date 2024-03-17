import torch
import numpy as np
from uncertainty_vit import EncoderViT

def loss(img, scale_factor, point_estimator, lower_estimator, upper_estimator):
    assert isinstance(point_estimator, EncoderViT)
    assert isinstance(lower_estimator, EncoderViT)
    assert isinstance(upper_estimator, EncoderViT)

    with torch.no_grad():
        # TODO: mask the data for lower and upper bounds (since they have uncertainty)? 
        # Possibly only need the masking in training quantile regression
        z_point = point_estimator(img).detach()
        z_lower = lower_estimator(img).detach()
        z_upper = upper_estimator(img).detach()

        assert len(z_point.shape) == 2
        assert z_point.shape == z_lower.shape == z_upper.shape

        success_set_count = 0
        total_items = z_point.shape[0] * z_point.shape[1]
        for i in range(z_point.shape[0]):
            for j in range(z_point.shape[1]):
                low = z_point[i, j] - scale_factor * (z_point[i, j] - z_lower[i, j])
                high = z_point[i, j] + scale_factor * (z_upper[i, j] - z_point[i, j])
                if z_point[i, j] >= low and z_point[i, j] <= high:
                    success_set_count += 1
        loss = 1 - success_set_count / total_items
        assert loss >= 0 and loss <= 1

    return loss

def calibrate(args, dataloader, 
              risk_level, error_rate, 
              point_estimator, lower_estimator, upper_estimator, 
              max_scale_factor, step_size):
    n = len(dataloader.dataset)
    scale_factor = max_scale_factor
    ucb = -1
    while ucb <= risk_level:
        scale_factor = scale_factor - step_size
        ucb = np.sqrt(1 / (2 * n) * np.log(1 / error_rate))
        for i, (img, label) in enumerate(dataloader):
            l_i = loss(img=img, scale_factor=scale_factor, 
                       point_estimator=point_estimator, 
                       lower_estimator=lower_estimator, upper_estimator=upper_estimator)
            ucb += l_i * label.shape[0] / n # scale loss by percentage of dataset it makes up
        assert ucb >= 0
    scale_factor = scale_factor + step_size # backtrack (loop ends on overshoot)

    return scale_factor

# TODO: use smth different from train or val set