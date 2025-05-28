#losses.py
"""
Loss functions for model training.
"""
import torch

def kl_divergence_loss(predicted_psf, target_psf, epsilon=1e-7):
    """
    Computes KL Divergence loss: sum(target * log(target / predicted)).
    Assumes inputs are probability distributions (sum to 1 spatially).

    Args:
        predicted_psf (torch.Tensor): Predicted map (B, 1, H, W). Output of Softmax.
        target_psf (torch.Tensor): Target map (B, 1, H, W). Should sum to 1 spatially.
        epsilon (float): Small value for numerical stability.

    Returns:
        torch.Tensor: Scalar KL divergence loss averaged over the batch.
    """
    # predicted_psf should already be output of softmax
    # Ensure target_psf is also a distribution (it should be from generate_single_psf)

    # Add epsilon to predicted for stability in log
    pred_clamped = torch.clamp(predicted_psf, min=epsilon)
    # Target should ideally not have zeros where prediction is non-zero,
    # but clamp target as well for robustness if needed.
    tgt_clamped = torch.clamp(target_psf, min=epsilon)

    # KL divergence: KL(Target || Predicted) = sum(Target * (log(Target) - log(Predicted)))
    # Note: log(Target) will be -inf where Target is 0. However, Target is 0 there,
    # so 0 * -inf = NaN. We only care about where Target > 0.
    # Let's compute where target > 0 to avoid log(0).
    kl_div = torch.where(target_psf > epsilon,
                         target_psf * (torch.log(tgt_clamped) - torch.log(pred_clamped)),
                         torch.zeros_like(target_psf))


    # Sum over spatial dimensions (H, W) and average over batch (B)
    # Need to handle potential NaN if target sums exactly to 0 (shouldn't happen with valid GT)
    kl_loss = kl_div.sum(dim=(2, 3)).mean()

    return kl_loss

# Example alternative: Simple L2 loss (Mean Squared Error)
# def mse_loss(predicted_psf, target_psf):
#     """Computes Mean Squared Error loss."""
#     loss = torch.mean((predicted_psf - target_psf) ** 2)
#     return loss
