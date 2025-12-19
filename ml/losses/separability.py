#!/usr/bin/env python3
"""
Separability Loss Module for Prime-X
=====================================
Fisher/LDA-style latent space separability optimization.

Purpose:
  Optimize for latent space class discrimination by maximizing:
    J = tr(S_B) / (tr(S_W) + eps)
  where S_B = between-class scatter, S_W = within-class scatter

  Then add to training loss as:
    loss_total = loss_vae - lambda * J
"""

from __future__ import annotations
import torch
import numpy as np


@torch.no_grad()
def _counts_per_class(y: torch.Tensor, n_classes: int = 3) -> list[int]:
    """Count samples per class. No gradients needed."""
    counts = []
    for c in range(n_classes):
        counts.append(int((y == c).sum().item()))
    return counts


def fisher_ratio_trace(
    z: torch.Tensor,
    y: torch.Tensor,
    n_classes: int = 3,
    eps: float = 1e-6,
    min_per_class: int = 8,
) -> torch.Tensor:
    """
    Fisher/LDA-style separability score.
    
    Computes: J = tr(S_B) / (tr(S_W) + eps)
    
    Args:
        z: (B, D) latent vectors, differentiable
        y: (B,) integer labels in {0, 1, ..., n_classes-1}
        n_classes: number of classes (default 3 for RM0/RM1)
        eps: numerical stability epsilon
        min_per_class: minimum samples per class; if violated, returns 0
    
    Returns:
        scalar Tensor, higher = better separation, differentiable w.r.t z
    
    Formula:
        tr(S_B) = sum_c n_c * ||mu_c - mu||^2
        tr(S_W) = sum_c sum_i ||z_i - mu_c||^2
        J = tr(S_B) / (tr(S_W) + eps)
    """
    # Reshape z to 2D if needed
    if z.ndim != 2:
        z = z.view(z.size(0), -1)
    
    # Ensure y is 1D long tensor
    y = y.view(-1).long()
    
    B, D = z.shape
    
    # PATCH-B: Use actual unique labels instead of assuming n_classes
    with torch.no_grad():
        unique_classes = torch.unique(y).long()
        n_classes_actual = int(unique_classes.max().item()) + 1
        
        # Build counts for ALL classes up to max
        counts_dict = {}
        for c in range(n_classes_actual):
            count = int((y == c).sum().item())
            counts_dict[c] = count
        
        nmin = min(counts_dict.values()) if counts_dict else 0
        if nmin < min_per_class:
            # Return 0 if any class has too few samples
            return z.new_tensor(0.0)
    
    # Compute traces (skip missing classes, always compute if possible)
    tr_sw = z.new_tensor(0.0)
    tr_sb = z.new_tensor(0.0)
    n_classes_present = 0
    
    for c in range(n_classes_actual):
        idx = (y == c)
        zc = z[idx]
        
        if zc.size(0) < min_per_class:
            # Skip this class if too few samples
            continue
        
        n_classes_present += 1
        muc = zc.mean(dim=0, keepdim=True)  # (1, D)
        
        # Within-class scatter trace: sum_i ||z_i - mu_c||^2
        tr_sw = tr_sw + ((zc - muc) ** 2).sum()
        
        # Between-class scatter trace: n_c * ||mu_c - mu||^2
        nc = float(zc.size(0))
        mu_global = z.mean(dim=0, keepdim=True)
        tr_sb = tr_sb + nc * ((muc - mu_global) ** 2).sum()
    
    # If no valid classes, return 0
    if n_classes_present == 0:
        return z.new_tensor(0.0)
    
    # Normalize by batch size for scale invariance
    tr_sw = tr_sw / float(B)
    tr_sb = tr_sb / float(B)
    
    # Fisher ratio (higher = better)
    return tr_sb / (tr_sw + eps)


def lambda_schedule(
    step: int,
    base_lambda: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """
    Schedule for separability loss weight lambda.
    
    Warmup: lambda = 0 for 'warmup_steps' (let recon stabilize)
    Ramp: lambda linearly increases from 0 to base_lambda over 'ramp_steps'
    
    Args:
        step: current training step
        base_lambda: target lambda value
        warmup_steps: steps with lambda=0
        ramp_steps: steps over which lambda ramps to base_lambda
    
    Returns:
        float in [0, base_lambda]
    """
    if base_lambda <= 0:
        return 0.0
    
    if step < warmup_steps:
        return 0.0
    
    # Ramp phase
    progress = (step - warmup_steps) / max(1, ramp_steps)
    progress = max(0.0, min(1.0, progress))
    
    return base_lambda * progress


@torch.no_grad()
def linear_probe_accuracy(
    z: torch.Tensor,
    y: torch.Tensor,
    n_classes: int = 3,
    train_frac: float = 0.8,
) -> float:
    """
    Sanity check: train a simple linear classifier on latent z with proper train/test split.
    
    This validates that Fisher/Mahalanobis improvements are *real*:
    If Fisher ↑ and Mahalanobis ↑ but accuracy ↓, separation is artifactual.
    
    CRITICAL FIX (2025-12-18):
    - Train/test split enforced (train_frac=0.8 by default)
    - Prevents data leakage: fitting and testing on SAME batch returns artificial 100%
    - Only valid if holdout test set shows high accuracy
    
    Args:
        z: (B, D) latent vectors
        y: (B,) integer labels
        n_classes: number of classes
        train_frac: fraction of data for training (rest for test)
    
    Returns:
        accuracy in [0, 1] on HELD-OUT test set, no gradient
    
    Implementation:
        Stratified train/test split → fit on train → score on test.
        Test accuracy is meaningful; train accuracy is ignored.
    """
    # Need at least 2 samples per class × 2 (train + test)
    if z.shape[0] < n_classes * 4:
        return 0.0
    
    # Ensure CPU for numpy/sklearn
    z_np = z.detach().cpu().numpy()  # (B, D)
    y_np = y.detach().cpu().numpy()  # (B,)
    
    # Filter out NaN/Inf values
    valid_mask = np.isfinite(z_np).all(axis=1) & np.isfinite(y_np)
    if not valid_mask.any() or valid_mask.sum() < n_classes * 4:
        return 0.0
    
    z_np = z_np[valid_mask]
    y_np = y_np[valid_mask]
    
    # DEBUG: Check label distribution
    unique_labels, counts = np.unique(y_np, return_counts=True)
    # Remap labels to 0,1,2,... for classifier compatibility
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    y_np_remapped = np.array([label_map[y] for y in y_np])
    
    try:
        import warnings
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        # Suppress sklearn warnings temporarily
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Stratified split ensures all classes present in both train and test
            z_train, z_test, y_train, y_test = train_test_split(
                z_np, y_np_remapped,
                train_size=train_frac,
                test_size=1.0 - train_frac,
                stratify=y_np_remapped,
                random_state=42,
            )
            
            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=100,
                multi_class="multinomial",
                random_state=42,
                tol=1e-3,
            )
            
            # FIT ON TRAIN, TEST ON TEST (the correct order!)
            clf.fit(z_train, y_train)
            accuracy = clf.score(z_test, y_test)  # Holdout test accuracy
            return float(accuracy)
    except Exception:
        # If fitting fails (e.g., singular matrix), return 0
        return 0.0
