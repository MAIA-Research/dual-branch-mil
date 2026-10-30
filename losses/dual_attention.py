import torch
import torch.nn.functional as F


def _cluster_fuse_heads(attns, header_attention):
    compute_dtype = torch.float32
    header_attention = torch.nan_to_num(
        header_attention.to(device=attns[0].device, dtype=compute_dtype),
        nan=0.0,
        neginf=-1e8,
        posinf=1e8,
    )
    head_weights = torch.softmax(header_attention, dim=0)
    fused = []
    for attn in attns:
        attn = torch.nan_to_num(
            attn.to(device=attns[0].device, dtype=compute_dtype),
            nan=0.0,
            neginf=0.0,
            posinf=1e8,
        )
        fused.append(torch.einsum('bhqt,h->bqt', attn, head_weights).squeeze(1))
    return fused


def _cluster_normalize(pred_attn, target):
    pred_attn = torch.nan_to_num(
        pred_attn.squeeze(), nan=0.0, neginf=0.0, posinf=1e8
    ).clamp_min(1e-8)
    target = torch.nan_to_num(
        target.squeeze(), nan=0.0, neginf=0.0, posinf=1e8
    ).clamp_min(0.0)
    if target.dim() != pred_attn.dim():
        target = target.unsqueeze(0) if pred_attn.dim() == 2 else target
    pred_attn = pred_attn / (pred_attn.sum(dim=-1, keepdim=True) + 1e-8)
    target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
    return pred_attn, target


def _cluster_rank_loss(pred_attn, target, ratio_margin):
    pred_flat = pred_attn.float().reshape(-1)
    target_flat = target.float().reshape(-1)
    if pred_flat.numel() > 1:
        pred_flat = pred_flat[1:]
        target_flat = target_flat[1:]

    high_idx = target_flat > torch.quantile(target_flat, 0.70)
    low_idx = target_flat < torch.quantile(target_flat, 0.30)
    if not high_idx.any():
        high_idx = target_flat > 1e-8
    if not low_idx.any():
        low_idx = target_flat <= 1e-8
    if not high_idx.any() or not low_idx.any():
        return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

    high_attn = pred_flat[high_idx].mean()
    low_attn = pred_flat[low_idx].mean()
    return F.relu(float(ratio_margin) - high_attn / low_attn.clamp_min(1e-8))


def _cluster_outside_attention_loss(pred_attn, target):
    outside_mask = target <= 1e-8
    if outside_mask.numel() > 0:
        outside_mask = outside_mask.clone()
        outside_mask[..., 0] = False
    if not outside_mask.any():
        return torch.zeros((), device=pred_attn.device, dtype=torch.float32)
    return pred_attn.masked_select(outside_mask).sum()


def _cluster_inside_attention_loss(pred_attn, target):
    inside_mask = target > 1e-8
    if inside_mask.numel() > 0:
        inside_mask = inside_mask.clone()
        inside_mask[..., 0] = False
    if not inside_mask.any():
        return torch.zeros((), device=pred_attn.device, dtype=torch.float32)
    return 1.0 - pred_attn.masked_select(inside_mask).sum().clamp(0.0, 1.0)


def cluster_attention_loss(
    attns,
    heatmap_0,
    heatmap_1,
    has_heatmap_0,
    has_heatmap_1,
    header_attention,
    rank_weight=1.0,
    rank_ratio_margin=1.5,
    kl_weight=0.0,
    l1_weight=0.0,
    outside_attention_weight=0.0,
    inside_attention_weight=0.0,
):
    """Selected-cluster rank loss used by TransMIL_attention_cluster."""
    if has_heatmap_0 == 0 and has_heatmap_1 == 0:
        return torch.tensor(0.0, device=heatmap_0.device, dtype=torch.float32)
    if not isinstance(attns, (list, tuple)) or len(attns) == 0:
        raise ValueError('attns must be a non-empty list or tuple of attention tensors')

    fused_attns = _cluster_fuse_heads(attns, header_attention)
    loss = torch.tensor(0.0, device=fused_attns[0].device, dtype=torch.float32)
    targets = ((heatmap_0, has_heatmap_0), (heatmap_1, has_heatmap_1))
    for layer_idx, (heatmap, has_heatmap) in enumerate(targets):
        if has_heatmap == 0:
            continue
        pred_attn, target = _cluster_normalize(
            fused_attns[layer_idx], heatmap.to(fused_attns[layer_idx].device)
        )
        rank = _cluster_rank_loss(pred_attn, target, rank_ratio_margin)
        outside = _cluster_outside_attention_loss(pred_attn, target)
        inside = _cluster_inside_attention_loss(pred_attn, target)
        layer_loss = (
            float(rank_weight) * rank
            + float(outside_attention_weight) * outside
            + float(inside_attention_weight) * inside
        )
        if kl_weight > 0:
            layer_loss = layer_loss + float(kl_weight) * F.kl_div(
                torch.log(pred_attn + 1e-8), target, reduction='sum'
            )
        if l1_weight > 0:
            layer_loss = layer_loss + float(l1_weight) * F.l1_loss(
                pred_attn, target, reduction='sum'
            )
        loss = loss + layer_loss

    if not torch.isfinite(loss):
        print('cluster_attention_loss became non-finite; returning 0.0 for this sample')
        return torch.zeros((), device=loss.device, dtype=torch.float32)
    return loss

# def attention_loss(attns, heatmap, has_heatmap, header_attention, head_fusion='attn', loss_fn='KL_div'):
#     if has_heatmap == 0:
#         device = heatmap.device if isinstance(heatmap, torch.Tensor) else torch.device('cpu')
#         return torch.tensor(0.0, device=device, dtype=torch.float32)

#     target = heatmap.squeeze()
#     if not isinstance(attns, (list, tuple)) or len(attns) == 0:
#         raise ValueError('attns must be a non-empty list or tuple of attention tensors')

#     attn_device = attns[0].device
#     compute_dtype = torch.float32

#     if head_fusion == 'max':
#         max_idx_header = torch.argmax(header_attention)
#         attns = [attn[:, max_idx_header, :, :].to(device=attn_device, dtype=compute_dtype) for attn in attns]
#     elif head_fusion == 'attn':
#         header_attention = torch.nan_to_num(
#             header_attention.to(device=attn_device, dtype=compute_dtype),
#             nan=0.0,
#             neginf=-1e8,
#             posinf=1e8,
#         )
#         header_attn = torch.softmax(header_attention, dim=0)
#         attns = [
#             torch.nan_to_num(
#                 attn.to(device=attn_device, dtype=compute_dtype),
#                 nan=0.0,
#                 neginf=-1e8,
#                 posinf=1e8,
#             )
#             for attn in attns
#         ]
#         attns = [torch.einsum('bijk,i->bjk', attn, header_attn).squeeze(1) for attn in attns]
#     else:
#         attns = [attn.to(device=attn_device, dtype=compute_dtype) for attn in attns]

#     pred_attn = attns[0].squeeze()
#     print("pred_attn shape:", pred_attn.shape)
#     print("target shape:", target.shape)
#     target = target.to(device=pred_attn.device, dtype=compute_dtype)

#     pred_attn = torch.nan_to_num(pred_attn, nan=0.0, neginf=0.0, posinf=1e8)
#     target = torch.nan_to_num(target, nan=0.0, neginf=0.0, posinf=1e8)
#     pred_attn = pred_attn.clamp_min(1e-8)
#     target = target.clamp_min(0.0)

#     if target.dim() != pred_attn.dim():
#         target = target.unsqueeze(0) if pred_attn.dim() == 2 else target

#     pred_sum = pred_attn.sum(dim=-1, keepdim=True) if pred_attn.dim() > 1 else pred_attn.sum()
#     target_sum = target.sum(dim=-1, keepdim=True) if target.dim() > 1 else target.sum()
#     pred_attn = pred_attn / (pred_sum + 1e-8)
#     target = target / (target_sum + 1e-8)

#     if loss_fn == 'l1':
#         loss = F.l1_loss(pred_attn, target, reduction='sum')
#     elif loss_fn == 'KL_div':
#         log_pred = torch.log(pred_attn + 1e-8)
#         loss = F.kl_div(log_pred, target, reduction='sum')
#     else:
#         raise ValueError(f"Unknown loss_fn: {loss_fn}")

#     if not torch.isfinite(loss):
#         print("attention_loss became non-finite; returning 0.0 for this sample")
#         return torch.zeros((), device=pred_attn.device, dtype=compute_dtype)

#     return loss

# def attention_loss(attns, heatmap_0, heatmap_1, has_heatmap_0, has_heatmap_1, header_attention, head_fusion='attn', loss_fn='KL_div'):
#     if has_heatmap_0 == 0 and has_heatmap_1 == 0:
#         device = heatmap_0.device if isinstance(heatmap_0, torch.Tensor) else torch.device('cpu')
#         return torch.tensor(0.0, device=device, dtype=torch.float32)

#     device = heatmap_0.device
#     loss_all = torch.tensor(0.0, device=device, dtype=torch.float32)

#     if not isinstance(attns, (list, tuple)) or len(attns) == 0:
#         raise ValueError('attns must be a non-empty list or tuple of attention tensors')

#     attn_device = attns[0].device
#     compute_dtype = torch.float32

#     if head_fusion == 'max':
#         max_idx_header = torch.argmax(header_attention)
#         attns = [attn[:, max_idx_header, :, :].to(device=attn_device, dtype=compute_dtype) for attn in attns]
#     elif head_fusion == 'attn':
#         header_attention = torch.nan_to_num(
#             header_attention.to(device=attn_device, dtype=compute_dtype),
#             nan=0.0,
#             neginf=-1e8,
#             posinf=1e8,
#         )
#         header_attn = torch.softmax(header_attention, dim=0)
#         attns = [
#             torch.nan_to_num(
#                 attn.to(device=attn_device, dtype=compute_dtype),
#                 nan=0.0,
#                 neginf=-1e8,
#                 posinf=1e8,
#             )
#             for attn in attns
#         ]
#         attns = [torch.einsum('bijk,i->bjk', attn, header_attn).squeeze(1) for attn in attns]
#     else:
#         attns = [attn.to(device=attn_device, dtype=compute_dtype) for attn in attns]

#     if has_heatmap_0 > 0:

#         target_0 = heatmap_0.squeeze()

#         pred_attn_0 = attns[0].squeeze()
#         target_0 = target_0.to(device=pred_attn_0.device, dtype=compute_dtype)

#         pred_attn_0 = torch.nan_to_num(pred_attn_0, nan=0.0, neginf=0.0, posinf=1e8)
#         target_0 = torch.nan_to_num(target_0, nan=0.0, neginf=0.0, posinf=1e8)
#         pred_attn_0 = pred_attn_0.clamp_min(1e-8)
#         target_0 = target_0.clamp_min(0.0)

#         if target_0.dim() != pred_attn_0.dim():
#             target_0 = target_0.unsqueeze(0) if pred_attn_0.dim() == 2 else target_0

#         pred_sum_0 = pred_attn_0.sum(dim=-1, keepdim=True) if pred_attn_0.dim() > 1 else pred_attn_0.sum()
#         target_sum_0 = target_0.sum(dim=-1, keepdim=True) if target_0.dim() > 1 else target_0.sum()
#         pred_attn_0 = pred_attn_0 / (pred_sum_0 + 1e-8)
#         target_0 = target_0 / (target_sum_0 + 1e-8)

#         if loss_fn == 'l1':
#             loss_all += F.l1_loss(pred_attn_0, target_0, reduction='sum')
#         elif loss_fn == 'KL_div':
#             log_pred_0 = torch.log(pred_attn_0 + 1e-8)
#             loss_all += F.kl_div(log_pred_0, target_0, reduction='sum')
#         else:
#             raise ValueError(f"Unknown loss_fn: {loss_fn}")


#     if has_heatmap_1 > 0:

#         target_1 = heatmap_1.squeeze()

#         pred_attn_1 = attns[1].squeeze()
#         target_1 = target_1.to(device=pred_attn_1.device, dtype=compute_dtype)

#         pred_attn_1 = torch.nan_to_num(pred_attn_1, nan=0.0, neginf=0.0, posinf=1e8)
#         target_1 = torch.nan_to_num(target_1, nan=0.0, neginf=0.0, posinf=1e8)
#         pred_attn_1 = pred_attn_1.clamp_min(1e-8)
#         target_1 = target_1.clamp_min(0.0)

#         if target_1.dim() != pred_attn_1.dim():
#             target_1 = target_1.unsqueeze(0) if pred_attn_1.dim() == 2 else target_1

#         pred_sum_1 = pred_attn_1.sum(dim=-1, keepdim=True) if pred_attn_1.dim() > 1 else pred_attn_1.sum()
#         target_sum_1 = target_1.sum(dim=-1, keepdim=True) if target_1.dim() > 1 else target_1.sum()
#         pred_attn_1 = pred_attn_1 / (pred_sum_1 + 1e-8)
#         target_1 = target_1 / (target_sum_1 + 1e-8)

#         if loss_fn == 'l1':
#             loss_all += F.l1_loss(pred_attn_1, target_1, reduction='sum')
#         elif loss_fn == 'KL_div':
#             log_pred_1 = torch.log(pred_attn_1 + 1e-8)
#             loss_all += F.kl_div(log_pred_1, target_1, reduction='sum')
#         else:
#             raise ValueError(f"Unknown loss_fn: {loss_fn}")

#     if not torch.isfinite(loss_all):
#         print("attention_loss became non-finite; returning 0.0 for this sample")
#         return torch.zeros((), device=loss_all.device, dtype=compute_dtype)

#     return loss_all


def attention_loss(
    attns,
    heatmap_0,
    heatmap_1,
    has_heatmap_0,
    has_heatmap_1,
    header_attention,
    head_fusion='attn',
    loss_fn='combined',
    alpha=10.0, ### 5.0
    beta=1.0,
    gamma=1.0,
    margin=0.1,
    rank_weight=1.0,
    outside_attention_weight=0.0,
    inside_attention_weight=0.0,
    outside_only=False,
    spike_regularization_weight=0.0,
    spike_max_p99_limit=50.0,
    spike_p99_p50_weight=0.0,
    spike_p99_p50_limit=20.0,
    spike_quantile=0.99,
    return_stats=False,
):
    if has_heatmap_0 == 0 and has_heatmap_1 == 0:
        device = heatmap_0.device if isinstance(heatmap_0, torch.Tensor) else torch.device('cpu')
        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        return (loss, {}) if return_stats else loss

    device = heatmap_0.device
    loss_all = torch.tensor(0.0, device=device, dtype=torch.float32)
    stats_all = []

    if not isinstance(attns, (list, tuple)) or len(attns) == 0:
        raise ValueError('attns must be a non-empty list or tuple of attention tensors')

    attn_device = attns[0].device
    compute_dtype = torch.float32

    if head_fusion == 'max':
        max_idx_header = torch.argmax(header_attention)
        attns = [attn[:, max_idx_header, :, :].to(device=attn_device, dtype=compute_dtype) for attn in attns]
    elif head_fusion == 'attn':
        header_attention = torch.nan_to_num(
            header_attention.to(device=attn_device, dtype=compute_dtype),
            nan=0.0,
            neginf=-1e8,
            posinf=1e8,
        )
        header_attn = torch.softmax(header_attention, dim=0)
        attns = [
            torch.nan_to_num(
                attn.to(device=attn_device, dtype=compute_dtype),
                nan=0.0,
                neginf=-1e8,
                posinf=1e8,
            )
            for attn in attns
        ]
        attns = [torch.einsum('bijk,i->bjk', attn, header_attn).squeeze(1) for attn in attns]
    else:
        attns = [attn.to(device=attn_device, dtype=compute_dtype) for attn in attns]

    def compute_attention_stats(pred_attn, target):
        inside_mask = target > 1e-8
        outside_mask = target <= 1e-8

        if inside_mask.numel() > 0:
            inside_mask = inside_mask.clone()
            outside_mask = outside_mask.clone()
            inside_mask[..., 0] = False
            outside_mask[..., 0] = False

        if not inside_mask.any() or not outside_mask.any():
            return {}

        pred_stats = pred_attn.float()
        inside_values = pred_stats.masked_select(inside_mask)
        outside_values = pred_stats.masked_select(outside_mask)
        flat = pred_stats.reshape(-1)

        inside_mean = inside_values.mean()
        outside_mean = outside_values.mean()
        p50 = torch.quantile(flat, 0.50)
        p99 = torch.quantile(flat, 0.99)
        max_val = flat.max()

        return {
            "inside_mean": inside_mean,
            "outside_mean": outside_mean,
            "inside_outside_ratio": inside_mean / outside_mean.clamp_min(1e-8),
            "inside_outside_diff": inside_mean - outside_mean,
            "max_p99_ratio": max_val / p99.clamp_min(1e-8),
            "p99_p50_ratio": p99 / p50.clamp_min(1e-8),
        }

    def merge_attention_stats(stats_list):
        stats_list = [stats for stats in stats_list if stats]
        if not stats_list:
            return {}
        merged = {}
        for key in stats_list[0]:
            values = [stats[key].to(dtype=torch.float32) for stats in stats_list if key in stats]
            if values:
                merged[key] = torch.stack(values).mean()
        return merged

    def compute_rank_loss(pred_attn, target):
        pred_flat = pred_attn.float().reshape(-1)
        target_flat = target.float().reshape(-1)

        high_idx = target_flat > torch.quantile(target_flat, 0.70)
        low_idx = target_flat < torch.quantile(target_flat, 0.30)

        if not high_idx.any() or not low_idx.any():
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        A_high = pred_flat[high_idx].mean()
        A_low = pred_flat[low_idx].mean()

        eps = torch.finfo(A_low.dtype).eps
        ratio = A_high / A_low.clamp_min(eps)

        print("high count", high_idx.sum().item(), "low count", low_idx.sum().item())
        print("A_high", A_high.item(), "A_low", A_low.item(), "ratio", (A_high / A_low.clamp_min(1e-8)).item())


        rank_loss = F.relu(1.5 - ratio) #### 2.0
        return rank_loss

    # def compute_rank_loss(pred_attn, target):
    #     pred_flat = pred_attn.reshape(-1)
    #     target_flat = target.reshape(-1)

    #     high_idx = target_flat > torch.quantile(target_flat, 0.7)
    #     low_idx = target_flat < torch.quantile(target_flat, 0.3)

    #     if not high_idx.any() or not low_idx.any():
    #         return torch.zeros((), device=pred_attn.device, dtype=pred_attn.dtype)

    #     # print(pred_flat.max())
    #     # print(pred_flat.min())
    #     # print(pred_flat.mean())

    #     # print(pred_flat[high_idx].mean())
    #     # print(pred_flat[low_idx].mean())

    #     # return F.relu(
    #     #     margin
    #     #     - pred_flat[high_idx].mean()
    #     #     + pred_flat[low_idx].mean()
    #     # )

    #     high_idx = target_flat > torch.quantile(target_flat, 0.90)
    #     low_idx  = target_flat < torch.quantile(target_flat, 0.10)

    #     A_high = pred_flat[high_idx].mean()
    #     A_low = pred_flat[low_idx].mean()

    #     print("high count", high_idx.sum().item(), "low count", low_idx.sum().item())
    #     print("A_high", A_high.item(), "A_low", A_low.item(), "ratio", (A_high / A_low.clamp_min(1e-8)).item())

    #     rank_loss = F.relu(2.0 - A_high / A_low.clamp_min(1e-4))

    #     return rank_loss

    def compute_outside_attention_loss(pred_attn, target):
        if outside_attention_weight <= 0:
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        outside_mask = target <= 1e-8
        if outside_mask.numel() > 0:
            outside_mask = outside_mask.clone()
            outside_mask[..., 0] = False

        if not outside_mask.any():
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        return pred_attn.masked_select(outside_mask).sum()

    def compute_inside_attention_loss(pred_attn, target):
        if inside_attention_weight <= 0:
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        inside_mask = target > 1e-8
        if inside_mask.numel() > 0:
            inside_mask = inside_mask.clone()
            inside_mask[..., 0] = False

        if not inside_mask.any():
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        inside_attention = pred_attn.masked_select(inside_mask).sum()
        return 1.0 - inside_attention.clamp(0.0, 1.0)

    def compute_spike_regularization(pred_attn):
        if spike_regularization_weight <= 0 and spike_p99_p50_weight <= 0:
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        flat = pred_attn.float().reshape(-1)
        if flat.numel() > 1:
            flat = flat[1:]
        if flat.numel() < 4:
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        q = min(max(float(spike_quantile), 0.50), 0.999)
        p99 = torch.quantile(flat, q).clamp_min(1e-8)
        p50 = torch.quantile(flat, 0.50).clamp_min(1e-8)
        max_p99_ratio = flat.max() / p99
        p99_p50_ratio = p99 / p50

        max_p99_penalty = F.relu(max_p99_ratio / max(float(spike_max_p99_limit), 1e-6) - 1.0).pow(2)
        p99_p50_penalty = F.relu(p99_p50_ratio / max(float(spike_p99_p50_limit), 1e-6) - 1.0).pow(2)
        spike_loss = spike_regularization_weight * max_p99_penalty + spike_p99_p50_weight * p99_p50_penalty

        if spike_loss.detach().item() > 0:
            print(
                f"spike_loss: {spike_loss.item():.4f}, "
                f"max/p99: {max_p99_ratio.item():.4f}, p99/p50: {p99_p50_ratio.item():.4f}"
            )
        return spike_loss

    def compute_attn_loss(pred_attn, target):
        outside = compute_outside_attention_loss(pred_attn, target)
        inside = compute_inside_attention_loss(pred_attn, target)
        spike = compute_spike_regularization(pred_attn)
        if outside_only:
            return outside_attention_weight * outside + inside_attention_weight * inside + spike
        rank = compute_rank_loss(pred_attn, target)
        if loss_fn == 'l1':
            return F.l1_loss(pred_attn, target, reduction='sum') + outside_attention_weight * outside + inside_attention_weight * inside + spike
        if loss_fn == 'KL_div':
            log_pred = torch.log(pred_attn + 1e-8)
            return alpha*F.kl_div(log_pred, target, reduction='sum') + outside_attention_weight * outside + inside_attention_weight * inside + spike
        if loss_fn == 'rank':
            return rank_weight * rank + outside_attention_weight * outside + inside_attention_weight * inside + spike
        if loss_fn == 'combined':
            log_pred = torch.log(pred_attn + 1e-8)
            kl = F.kl_div(log_pred, target, reduction='sum')
            l1 = F.l1_loss(pred_attn, target, reduction='sum')
            cosine = F.cosine_similarity(
                pred_attn.reshape(1, -1),
                target.reshape(1, -1),
                dim=1,
                eps=1e-8,
            ).squeeze()
            print(
                f"kl: {kl.item():.4f}, l1: {l1.item():.4f}, "
                f"cosine_similarity: {cosine.item():.4f}, rank: {rank.item():.4f}, "
                f"outside_attention: {outside.item():.4f}, inside_loss: {inside.item():.4f}"
            )
            return (
                alpha * kl
                + beta * l1
                + gamma * (1.0 - cosine)
                + rank_weight * rank
                + outside_attention_weight * outside
                + inside_attention_weight * inside
                + spike
            )

            # print(
            #     f"kl: {kl.item():.4f}, "
            #     f"rank: {rank.item():.4f}"
            # )
            # return alpha * kl + rank_weight * rank
        raise ValueError(f"Unknown loss_fn: {loss_fn}")

    if has_heatmap_0 > 0:

        target_0 = heatmap_0.squeeze()

        pred_attn_0 = attns[0].squeeze()
        target_0 = target_0.to(device=pred_attn_0.device, dtype=compute_dtype)

        pred_attn_0 = torch.nan_to_num(pred_attn_0, nan=0.0, neginf=0.0, posinf=1e8)
        target_0 = torch.nan_to_num(target_0, nan=0.0, neginf=0.0, posinf=1e8)
        pred_attn_0 = pred_attn_0.clamp_min(1e-8)
        target_0 = target_0.clamp_min(0.0)

        if target_0.dim() != pred_attn_0.dim():
            target_0 = target_0.unsqueeze(0) if pred_attn_0.dim() == 2 else target_0

        pred_sum_0 = pred_attn_0.sum(dim=-1, keepdim=True) if pred_attn_0.dim() > 1 else pred_attn_0.sum()
        target_sum_0 = target_0.sum(dim=-1, keepdim=True) if target_0.dim() > 1 else target_0.sum()
        pred_attn_0 = pred_attn_0 / (pred_sum_0 + 1e-8)
        target_0 = target_0 / (target_sum_0 + 1e-8)

        stats_all.append(compute_attention_stats(pred_attn_0, target_0))
        loss_all += compute_attn_loss(pred_attn_0, target_0)


    if has_heatmap_1 > 0:

        target_1 = heatmap_1.squeeze()

        pred_attn_1 = attns[1].squeeze()
        target_1 = target_1.to(device=pred_attn_1.device, dtype=compute_dtype)

        pred_attn_1 = torch.nan_to_num(pred_attn_1, nan=0.0, neginf=0.0, posinf=1e8)
        target_1 = torch.nan_to_num(target_1, nan=0.0, neginf=0.0, posinf=1e8)
        pred_attn_1 = pred_attn_1.clamp_min(1e-8)
        target_1 = target_1.clamp_min(0.0)

        if target_1.dim() != pred_attn_1.dim():
            target_1 = target_1.unsqueeze(0) if pred_attn_1.dim() == 2 else target_1

        pred_sum_1 = pred_attn_1.sum(dim=-1, keepdim=True) if pred_attn_1.dim() > 1 else pred_attn_1.sum()
        target_sum_1 = target_1.sum(dim=-1, keepdim=True) if target_1.dim() > 1 else target_1.sum()
        pred_attn_1 = pred_attn_1 / (pred_sum_1 + 1e-8)
        target_1 = target_1 / (target_sum_1 + 1e-8)

        stats_all.append(compute_attention_stats(pred_attn_1, target_1))
        loss_all += compute_attn_loss(pred_attn_1, target_1)

    if not torch.isfinite(loss_all):
        print("attention_loss became non-finite; returning 0.0 for this sample")
        loss_all = torch.zeros((), device=loss_all.device, dtype=compute_dtype)

    stats = merge_attention_stats(stats_all)
    if stats:
        print(
            "attn stats:",
            f"inside_mean={stats['inside_mean'].item():.8f}",
            f"outside_mean={stats['outside_mean'].item():.8f}",
            f"inside/outside={stats['inside_outside_ratio'].item():.4f}",
            f"inside-outside={stats['inside_outside_diff'].item():.8f}",
            f"max/p99={stats['max_p99_ratio'].item():.4f}",
            f"p99/p50={stats['p99_p50_ratio'].item():.4f}",
        )

    return (loss_all, stats) if return_stats else loss_all
