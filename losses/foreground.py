import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from .complementary import scale_gradient
from .union import head_scores, membership


def hybrid_asinh_normalize(attention, value_weight=0.2, scale=2.0):
    x = np.asarray(attention, dtype=np.float64)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    robust_scale = 1.4826 * mad + 1e-12
    transformed = np.arcsinh((x - median) / (scale * robust_scale))
    low, high = np.percentile(transformed, [1, 99])
    value_score = np.clip((transformed - low) / (high - low + 1e-12), 0, 1)
    distribution_score = (rankdata(x, method='average') - 1) / max(len(x) - 1, 1)
    return value_weight * value_score + (1 - value_weight) * distribution_score


@torch.no_grad()
def three_class_foreground_mask(attns, h, classifier, n_patches, threshold=0.4):
    if len(attns) != 2 or n_patches < 1 or classifier.out_features != 3:
        raise ValueError('Foreground selection requires two three-class layers and a nonempty bag')
    if not 0 <= threshold <= 1:
        raise ValueError('foreground_threshold must belong to [0, 1]')
    heads = attns[0].shape[1]
    with torch.autocast(device_type=h.device.type, enabled=False):
        header = head_scores(h.detach(), classifier, class_idx=2, n_heads=heads).softmax(0)
        result = torch.ones((1, n_patches + 1), device=h.device, dtype=torch.float32)
        for attn in attns:
            if tuple(attn.shape) != (1, heads, 1, n_patches + 1):
                raise ValueError(f'Foreground attention/patch mismatch: {tuple(attn.shape)}, N={n_patches}')
            fused = torch.einsum('bhqt,h->bqt', attn.detach().float(), header).squeeze(1)
            fused = fused.softmax(dim=1)
            fused = fused / fused.mean()
            result = ((fused * result) + result) / 2
    scores = result[0, 1:]
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Non-finite three-class foreground attention')
    span = scores.max() - scores.min()
    normalized = (scores - scores.min()) / span if span.item() > 0 else torch.zeros_like(scores)
    # Only the detached teacher vector crosses to CPU for the visualization's
    # NumPy median/percentiles and average-tie ranks; trainable dual scores stay on device.
    normalized = hybrid_asinh_normalize(normalized.cpu().numpy())
    return torch.as_tensor(normalized >= threshold, device=h.device, dtype=torch.bool)


def normalized_dual_attention(scores):
    return membership(scores, 'max')


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def validate_inputs(a0, a1, mask):
    if a0.ndim != 1 or a0.shape != a1.shape or a0.shape != mask.shape or a0.numel() == 0:
        raise ValueError('Foreground losses require aligned nonempty patch vectors and mask')
    if mask.dtype != torch.bool or mask.device != a0.device or a1.device != a0.device:
        raise ValueError('Foreground mask must be boolean and on the attention device')


def foreground_complementary_loss(a0, a1, mask, mode='symmetric', grad_weight_0=1.0, grad_weight_1=1.0):
    validate_inputs(a0, a1, mask)
    if mode not in {'symmetric', 'asymmetric'}:
        raise ValueError('complementary_mode must be symmetric or asymmetric')
    if not 0 <= grad_weight_0 <= 1 or not 0 <= grad_weight_1 <= 1:
        raise ValueError('Complementary gradient weights must belong to [0, 1]')
    a0_comp = scale_gradient(a0, grad_weight_0)
    a1_comp = scale_gradient(a1.detach() if mode == 'asymmetric' else a1, grad_weight_1)
    return masked_mean(a0_comp * a1_comp, mask)


def foreground_anticorrelation_loss(a0, a1, mask, rho_target=-0.3, eps=1e-8,
                                   mode='symmetric', grad_weight_0=1.0, grad_weight_1=1.0):
    validate_inputs(a0, a1, mask)
    if not np.isfinite(rho_target) or not -1 <= rho_target <= 1:
        raise ValueError('foreground_comp_rho_target must belong to [-1, 1]')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if mode not in {'symmetric', 'asymmetric'}:
        raise ValueError('complementary_mode must be symmetric or asymmetric')
    if not 0 <= grad_weight_0 <= 1 or not 0 <= grad_weight_1 <= 1:
        raise ValueError('Complementary gradient weights must belong to [0, 1]')
    x = scale_gradient(a0, grad_weight_0)[mask]
    y = scale_gradient(a1.detach() if mode == 'asymmetric' else a1, grad_weight_1)[mask]
    zero = (x.sum() + y.sum()) * 0.0
    if x.numel() < 2:
        return zero, None
    x, y = x - x.mean(), y - y.mean()
    norm0, norm1 = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    if (norm0.detach() < eps) or (norm1.detach() < eps):
        return zero, None
    rho = ((x * y).sum() / (norm0 * norm1).clamp_min(eps)).clamp(-1, 1)
    return F.relu(rho - rho_target), rho


def background_suppression_loss(a0, a1, mask, threshold=0.2):
    validate_inputs(a0, a1, mask)
    if not 0 <= threshold <= 1:
        raise ValueError('background_attention_threshold must belong to [0, 1]')
    background = ~mask
    return (masked_mean(F.relu(a0 - threshold).square(), background)
            + masked_mean(F.relu(a1 - threshold).square(), background)) / 2


def region_attention_statistics(a0, a1, mask):
    return {
        'fg_attention_branch0': masked_mean(a0, mask),
        'fg_attention_branch1': masked_mean(a1, mask),
        'bg_attention_branch0': masked_mean(a0, ~mask),
        'bg_attention_branch1': masked_mean(a1, ~mask),
    }
