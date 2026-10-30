import torch
import torch.nn.functional as F
import math


def head_scores(h, classifier, class_idx, n_heads):
    if h.shape != (1, classifier.in_features):
        raise ValueError(f"Unexpected pooled feature shape: {tuple(h.shape)}")
    if classifier.in_features % n_heads:
        raise ValueError("Pooled feature dimension must be divisible by attention heads")
    contributions = h.float()[0] * classifier.weight.detach().float()[class_idx]
    return contributions.reshape(n_heads, -1).sum(-1)


def patch_attention(attn, h, classifier, class_idx, n_patches):
    if attn.ndim != 4 or attn.shape[0] != 1 or attn.shape[2] != 1:
        raise ValueError(f"Expected [1, heads, 1, N+1] attention, got {tuple(attn.shape)}")
    if attn.shape[-1] != n_patches + 1:
        raise ValueError(f"Expected {n_patches + 1} tokens including CLS, got {attn.shape[-1]}")
    weights = head_scores(h, classifier, class_idx, attn.shape[1]).softmax(0)
    scores = torch.einsum('bhqt,h->bqt', attn.float(), weights)[0, 0, 1:]
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite patch attention")
    return scores.clamp_min(0)


def distribution(scores):
    scores = scores.float().clamp_min(1e-12)
    return scores / scores.sum()


def membership(scores, normalization):
    scores = scores.float().clamp_min(0)
    if normalization == 'max':
        return scores / scores.max().clamp_min(1e-12)
    if normalization == 'minmax':
        return (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-12)
    if normalization == 'sum':
        return distribution(scores)
    raise ValueError(f"Unknown union normalization: {normalization}")


def union_distribution(p0, p1, normalization='max'):
    if p0.ndim != 1 or p0.shape != p1.shape or p0.numel() == 0:
        raise ValueError("Union requires aligned, nonempty patch vectors")
    a = membership(p0, normalization)
    b = membership(p1, normalization)
    # Algebraically equal to 1-(1-a)*(1-b), with less cancellation near zero.
    return distribution(a + b - a * b)


def union_alignment(p0, p1, target, normalization='max', detach_target=True):
    if target.shape != p0.shape:
        raise ValueError("Three-class and dual patch vectors must have the same shape")
    union = union_distribution(p0, p1, normalization)
    target = distribution(target.detach() if detach_target else target)
    return F.kl_div(union.log(), target, reduction='sum').clamp_min(0)


def apply_union_margin(divergence, margin=0.0):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Union margin must be finite and nonnegative')
    return divergence if margin == 0 else F.relu(divergence - margin)
