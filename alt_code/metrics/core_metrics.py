# metrics/core_metrics.py

import torch
import numpy as np
from scipy.stats import pearsonr

def get_tensor_derivatives(A):
    """
    Calculates strain-rate tensor (s), rotation-rate tensor (w),
    magnitude of A (epsilon), and invariants q, r for velocity gradient A.
    Args:
        A: (batch, 3, 3) velocity gradient tensor
    Returns:
        s: (batch, 3, 3) symmetric part (strain-rate)
        w: (batch, 3, 3) antisymmetric part (rotation-rate)
        epsilon: (batch,) norm of A
        q: (batch,) 2nd invariant of normalized A
        r: (batch,) 3rd invariant of normalized A
    """
    s = 0.5 * (A + A.transpose(1, 2))
    w = 0.5 * (A - A.transpose(1, 2))
    epsilon_sq = torch.sum(A * A, dim=(1, 2), keepdim=True)
    epsilon = torch.sqrt(epsilon_sq)
    epsilon[epsilon == 0] = 1e-8
    a = A / epsilon
    # 2nd and 3rd invariants for incompressible flow
    a_sq = torch.einsum('bik,bkj->bij', a, a)
    q = -0.5 * torch.einsum('bii->b', a_sq)
    a_cubed = torch.einsum('bik,bkj->bij', a_sq, a)
    r = -1./3. * torch.einsum('bii->b', a_cubed)
    return s, w, epsilon.squeeze(), q, r

def process_ground_truth_Q(Q, epsilon):
    """
    Normalize the ground-truth Q tensor per paper.
    Args:
        Q: (batch, 3, 3) anisotropic pressure Hessian
        epsilon: (batch,) norm of velocity gradient A
    Returns:
        Q_prime: Q / epsilon^2
        Q_hat_prime: Q_prime normalized to unit norm
        psi: norm of Q_prime (scalar, per sample)
    """
    epsilon_sq = (epsilon ** 2).unsqueeze(-1).unsqueeze(-1)
    Q_prime = Q / (epsilon_sq + 1e-8)
    psi_sq = torch.sum(Q_prime * Q_prime, dim=(1, 2), keepdim=True)
    psi = torch.sqrt(psi_sq)
    psi[psi == 0] = 1e-8
    Q_hat_prime = Q_prime / psi
    return Q_prime, Q_hat_prime, psi.squeeze()


def euler_angle_loss_detailed(Q_hat_prime_pred, Q_hat_prime_true, s_true):
    """
    Compute the Euler angle loss (Eq. 20 of AIBM paper) and track individual components.

    Args:
        Q_hat_prime_pred: (batch, 3, 3) predicted direction tensor
        Q_hat_prime_true: (batch, 3, 3) ground truth direction tensor
        s_true:           (batch, 3, 3) strain-rate tensor (normalized, symmetric, traceless)

    Returns:
        loss_J: torch scalar loss (sum of 3 angle errors)
        components: dict with breakdown of angle errors and mean cosines
    """
    # Ensure CPU for eigen
    Q_hat_prime_pred = Q_hat_prime_pred.detach().cpu()
    Q_hat_prime_true = Q_hat_prime_true.detach().cpu()
    s_true = s_true.detach().cpu()

    # Eigenvectors of strain-rate tensor
    _, e_s_vecs = torch.linalg.eigh(s_true)
    e_gamma_s, e_beta_s, e_alpha_s = e_s_vecs[:, :, 0], e_s_vecs[:, :, 1], e_s_vecs[:, :, 2]

    # Eigenvectors of true Q'
    _, e_p_vecs_true = torch.linalg.eigh(Q_hat_prime_true)
    e_gamma_p_true, e_beta_p_true, e_alpha_p_true = e_p_vecs_true[:, :, 0], e_p_vecs_true[:, :, 1], e_p_vecs_true[:, :, 2]

    # Eigenvectors of predicted Q'
    _, e_p_vecs_pred = torch.linalg.eigh(Q_hat_prime_pred)
    e_gamma_p_pred, e_beta_p_pred, e_alpha_p_pred = e_p_vecs_pred[:, :, 0], e_p_vecs_pred[:, :, 1], e_p_vecs_pred[:, :, 2]

    # --- Calculate Euler angles for TRUE tensor ---
    cos_zeta_true = torch.einsum('bi,bi->b', e_gamma_p_true, e_gamma_s)
    e_proj_prime_true = e_alpha_p_true - torch.einsum('bi,bi->b', e_alpha_p_true, e_gamma_s).unsqueeze(1) * e_gamma_s
    e_proj_true = e_proj_prime_true / (torch.norm(e_proj_prime_true, dim=1, keepdim=True) + 1e-8)
    cos_theta_true = torch.einsum('bi,bi->b', e_alpha_s, e_proj_true)
    e_norm_true = torch.cross(e_proj_true, e_gamma_s)
    cos_eta_true = torch.einsum('bi,bi->b', e_beta_p_true, e_norm_true)

    # --- Calculate Euler angles for PREDICTED tensor ---
    cos_zeta_pred = torch.einsum('bi,bi->b', e_gamma_p_pred, e_gamma_s)
    e_proj_prime_pred = e_alpha_p_pred - torch.einsum('bi,bi->b', e_alpha_p_pred, e_gamma_s).unsqueeze(1) * e_gamma_s
    e_proj_pred = e_proj_prime_pred / (torch.norm(e_proj_prime_pred, dim=1, keepdim=True) + 1e-8)
    cos_theta_pred = torch.einsum('bi,bi->b', e_alpha_s, e_proj_pred)
    e_norm_pred = torch.cross(e_proj_pred, e_gamma_s)
    cos_eta_pred = torch.einsum('bi,bi->b', e_beta_p_pred, e_norm_pred)

    # --- Compute the three-part loss J (Eq. 20) ---
    L1 = torch.sum((torch.abs(cos_zeta_true) - torch.abs(cos_zeta_pred))**2) / (torch.sum(cos_zeta_true**2) + 1e-8)
    L2 = torch.sum((torch.abs(cos_theta_true) - torch.abs(cos_theta_pred))**2) / (torch.sum(cos_theta_true**2) + 1e-8)
    L3 = torch.sum((torch.abs(cos_eta_true) - torch.abs(cos_eta_pred))**2) / (torch.sum(cos_eta_true**2) + 1e-8)
    loss_J = L1 + L2 + L3

    # Return detailed metrics
    components = {
        'L1_zeta': L1.item(),
        'L2_theta': L2.item(),
        'L3_eta': L3.item(),
        'cos_zeta_true_mean': torch.abs(cos_zeta_true).mean().item(),
        'cos_zeta_pred_mean': torch.abs(cos_zeta_pred).mean().item(),
        'cos_theta_true_mean': torch.abs(cos_theta_true).mean().item(),
        'cos_theta_pred_mean': torch.abs(cos_theta_pred).mean().item(),
        'cos_eta_true_mean': torch.abs(cos_eta_true).mean().item(),
        'cos_eta_pred_mean': torch.abs(cos_eta_pred).mean().item(),
    }
    return loss_J.to(Q_hat_prime_pred.device), components

def magnitude_loss(psi_pred, psi_true):
    """
    Relative mean square error for the magnitude psi (Eq. 22 of the AIBM paper).
    Args:
        psi_pred: (batch,) predicted magnitude
        psi_true: (batch,) ground-truth magnitude
    Returns:
        torch scalar (relative MSE)
    """
    numerator = torch.sum((psi_true - psi_pred) ** 2)
    denominator = torch.sum(psi_true ** 2) + 1e-8
    return numerator / denominator

def basic_similarity_metrics(pred, true):
    """
    Compute batchwise Pearson and normalized cross-correlation (NCC) between predictions and targets.
    Args:
        pred: (batch, ...) prediction tensor
        true: (batch, ...) ground-truth tensor
    Returns:
        dict with mean/std for Pearson and NCC
    """
    pred_flat = pred.view(pred.shape[0], -1).cpu().numpy()
    true_flat = true.view(true.shape[0], -1).cpu().numpy()

    ncc_scores = []
    pearson_scores = []
    for i in range(pred_flat.shape[0]):
        p, t = pred_flat[i], true_flat[i]
        if np.std(p) > 1e-8 and np.std(t) > 1e-8:
            # Pearson
            corr, _ = pearsonr(p, t)
            pearson_scores.append(corr)
            # NCC (same as Pearson for mean-centered, normalized data)
            ncc = np.corrcoef(p, t)[0, 1]
            ncc_scores.append(ncc if not np.isnan(ncc) else 0.0)
        else:
            pearson_scores.append(0.0)
            ncc_scores.append(0.0)
    return {
        'pearson_mean': float(np.mean(pearson_scores)),
        'pearson_std': float(np.std(pearson_scores)),
        'ncc_mean': float(np.mean(ncc_scores)),
        'ncc_std': float(np.std(ncc_scores))
    }

def angle_means(Q_hat_prime_pred, Q_hat_prime_true, s_true):
    """
    Return mean cosine angles for reporting (zeta, theta, eta).
    """
    # Get only means for quick stats
    _, comps = euler_angle_loss_detailed(Q_hat_prime_pred, Q_hat_prime_true, s_true)
    return {
        'cos_zeta_true_mean': comps['cos_zeta_true_mean'],
        'cos_zeta_pred_mean': comps['cos_zeta_pred_mean'],
        'cos_theta_true_mean': comps['cos_theta_true_mean'],
        'cos_theta_pred_mean': comps['cos_theta_pred_mean'],
        'cos_eta_true_mean': comps['cos_eta_true_mean'],
        'cos_eta_pred_mean': comps['cos_eta_pred_mean'],
    }
