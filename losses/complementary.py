import torch.nn.functional as F

from .union import distribution


def scale_gradient(x, scale):
    return x.detach() + scale * (x - x.detach())


def complementary_loss(attn0, attn1, mode='symmetric', grad_weight_0=1.0, grad_weight_1=1.0):
    if mode not in {'symmetric', 'asymmetric'}:
        raise ValueError('complementary_mode must be symmetric or asymmetric')
    if not 0.0 <= grad_weight_0 <= 1.0 or not 0.0 <= grad_weight_1 <= 1.0:
        raise ValueError('Complementary gradient weights must belong to [0, 1]')
    # Preserve the existing asymmetric anchor regardless of its configured scale.
    attn1_for_comp = attn1.detach() if mode == 'asymmetric' else attn1
    attn0_comp = scale_gradient(attn0, grad_weight_0)
    attn1_comp = scale_gradient(attn1_for_comp, grad_weight_1)
    return F.cosine_similarity(
        distribution(attn0_comp)[None], distribution(attn1_comp)[None], dim=1,
    ).squeeze()
