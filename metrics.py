import torch


def compute_SADE_SFDE(norm, mask=None):
    '''
    Helper function to compute SADE and SFDE
    norm: [B, modes, T, N]
    mask: [B, T, N]
    '''
    if mask is None:
        mask = torch.ones_like(norm)
    else:
        mask[mask == 0] = torch.nan
        mask = mask.unsqueeze(1).repeat(1, norm.shape[1], 1, 1)
    assert norm.shape == mask.shape, f"norm shape {norm.shape} and mask shape {mask.shape} must be the same"
    # SADE: Sum of Average Displacement Error
    sade_min = torch.nanmean(norm * mask, dim=(2, 3)).min(dim=1).values
    sade_avg = torch.nanmean(norm * mask, dim=(2, 3)).mean(dim=1)

    # SFDE: Sum of Final Displacement Error
    sfde_min = torch.nanmean(norm[:, :, -1] * mask[:, :, -1], dim=(2)).min(dim=1).values
    sfde_avg = torch.nanmean(norm[:, :, -1] * mask[:, :, -1], dim=(2)).mean(dim=1)

    return {
        "sade_min": sade_min,  # [B]
        "sfde_min": sfde_min,  # [B]
        "sade_avg": sade_avg,  # [B]
        "sfde_avg": sfde_avg   # [B]
    }


def compute_ADE_FDE(norm, mask=None):
    '''
    Helper function to compute ADE and FDE
    norm: [B, modes, T, N]
    mask: [B, T, N]
    '''
    if mask is None:
        mask = torch.ones_like(norm)
    else:
        mask[mask == 0] = torch.nan
        mask = mask.unsqueeze(1).repeat(1, norm.shape[1], 1, 1)
    assert norm.shape == mask.shape, f"norm shape {norm.shape} and mask shape {mask.shape} must be the same"
    # ADE: Average Displacement Error
    ade_min = torch.nanmean(norm * mask, dim=(2)).min(dim=1).values.reshape(-1)  # [B * N]
    ade_avg = torch.nanmean(norm * mask, dim=(2)).mean(dim=1).reshape(-1)  # [B * N]

    # FDE: Final Displacement Error
    fde_min = (norm[:, :, -1] * mask[:, :, -1]).min(dim=1).values.reshape(-1)  # [B * N]
    fde_avg = (norm[:, :, -1] * mask[:, :, -1]).mean(dim=1).reshape(-1)  # [B * N]

    return {
        "ade_min": ade_min,  # [B * N]
        "fde_min": fde_min,  # [B * N]
        "ade_avg": ade_avg,  # [B * N]
        "fde_avg": fde_avg   # [B * N]
    }


def compute_ACC(pred, gt, mask, unsqueeze_gt=True):
    """
    Computes accuracy of possession prediction.

    Args:
        pred (torch.Tensor): [B, modes, T] tensor of integer event types (pred)
        gt (torch.Tensor): [B, T] tensor of integer event types (gt)
        mask (torch.Tensor): [B, T] tensor of boolean values indicating prediction mask.
    Returns:
        torch.Tensor: [B] tensor of accuracy.
    """
    if mask is None:
        mask = torch.ones_like(pred)
    else:
        mask[mask == 0] = torch.nan
        mask = mask.unsqueeze(1).repeat(1, pred.shape[1], 1)
    assert pred.shape == mask.shape, f"pred shape {pred.shape} and mask shape {mask.shape} must be the same"
    if unsqueeze_gt:
        gt = gt.unsqueeze(1)
    acc = (pred == gt).float() * mask
    
    temporal_acc = torch.nanmean(acc, dim=-1)  # [B, modes]
    
    acc_max = torch.max(temporal_acc, dim=1)[0]
    acc_mean = torch.mean(temporal_acc, dim=1)
    acc_min = torch.min(temporal_acc, dim=1)[0]

    return {
        "acc_min": acc_min,  # [B]
        "acc_max": acc_max,  # [B]
        "acc_mean": acc_mean  # [B]
    }
