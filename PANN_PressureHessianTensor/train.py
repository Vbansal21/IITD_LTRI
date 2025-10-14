import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime, pathlib
import scipy.io
import pandas as pd
import os
import argparse
import glob
import collections
from pathlib import Path
from typing import Optional

# --- Put once at top of your module (before plt imports) ---
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for servers/CI

plt.rcParams.update({
    "savefig.dpi": 600,
    "figure.autolayout": True,
    "font.size": 13,
    "text.usetex": False,           # use MathText, avoid external LaTeX
    "mathtext.fontset": "stix"      # stable math font
})

# --- Configuration ---
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. UTILITY FUNCTIONS FOR TENSOR OPERATIONS
# ==============================================================================

def get_tensor_derivatives(A):
    s = 0.5 * (A + A.transpose(1, 2))
    w = 0.5 * (A - A.transpose(1, 2))
    epsilon_sq = torch.sum(A * A, dim=(1, 2), keepdim=True)
    epsilon = torch.sqrt(epsilon_sq + 1e-16)
    a = A / epsilon
    a_sq = torch.einsum('bik,bkj->bij', a, a)
    q = -0.5 * torch.einsum('bii->b', a_sq)
    a_cubed = torch.einsum('bik,bkj->bij', a_sq, a)
    r = -1./3. * torch.einsum('bii->b', a_cubed)
    return s, w, epsilon.squeeze(), q, r

def process_ground_truth_Q(Q, epsilon):
    epsilon_sq = (epsilon**2).unsqueeze(-1).unsqueeze(-1)
    Q_prime = Q / epsilon_sq
    psi_sq = torch.sum(Q_prime * Q_prime, dim=(1, 2), keepdim=True)
    psi = torch.sqrt(psi_sq + 1e-16)
    Q_hat_prime = Q_prime / psi
    return Q_prime, Q_hat_prime, psi.squeeze()

def normalize_strain_tensor(A, epsilon):
    """Return s = sym(a) with a = A / ||A||, guarding against 0-norm tensors."""
    eps_matrix = epsilon.unsqueeze(-1).unsqueeze(-1)
    a_norm = A / (eps_matrix + 1e-16)
    s_norm = 0.5 * (a_norm + a_norm.transpose(1, 2))
    return s_norm

def compute_tbnn_invariants_and_bases(s_input, w_input):
    """Compute scalar invariants λ₁…λ₅ and tensor bases T₁…T₁₀ for the TBNN."""
    s_sq = torch.matmul(s_input, s_input)
    w_sq = torch.matmul(w_input, w_input)
    lambda_1 = torch.diagonal(s_sq, dim1=-2, dim2=-1).sum(-1)
    lambda_2 = torch.diagonal(w_sq, dim1=-2, dim2=-1).sum(-1)
    s_cubed = torch.matmul(s_sq, s_input)
    lambda_3 = torch.diagonal(s_cubed, dim1=-2, dim2=-1).sum(-1)
    w_sq_s = torch.matmul(w_sq, s_input)
    lambda_4 = torch.diagonal(w_sq_s, dim1=-2, dim2=-1).sum(-1)
    w_sq_s_sq = torch.matmul(w_sq, s_sq)
    lambda_5 = torch.diagonal(w_sq_s_sq, dim1=-2, dim2=-1).sum(-1)
    invariants = torch.stack([lambda_1, lambda_2, lambda_3, lambda_4, lambda_5], dim=1)

    eye = torch.eye(3, device=s_input.device, dtype=s_input.dtype).unsqueeze(0).expand_as(s_input)
    T1 = s_input
    T2 = torch.matmul(s_input, w_input) - torch.matmul(w_input, s_input)
    T3 = s_sq - torch.diagonal(s_sq, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1) / 3.0 * eye
    T4 = w_sq - torch.diagonal(w_sq, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1) / 3.0 * eye
    T5 = torch.matmul(w_input, s_sq) - torch.matmul(s_sq, w_input)
    sw2 = torch.matmul(s_input, w_sq)
    w2s = torch.matmul(w_sq, s_input)
    T6 = w2s + sw2 - 2.0 / 3.0 * torch.diagonal(sw2, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1) * eye
    ws = torch.matmul(w_input, s_input)
    sw = torch.matmul(s_input, w_input)
    T7 = torch.matmul(ws, w_sq) - torch.matmul(w_sq, sw)
    s2w = torch.matmul(s_sq, w_input)
    T8 = torch.matmul(sw, s_sq) - torch.matmul(s_sq, ws)
    w2s2 = torch.matmul(w_sq, s_sq)
    s2w2 = torch.matmul(s_sq, w_sq)
    T9 = w2s2 + s2w2 - 2.0 / 3.0 * torch.diagonal(s2w2, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1) * eye
    ws2 = torch.matmul(w_input, s_sq)
    T10 = torch.matmul(ws2, w_sq) - torch.matmul(w_sq, s2w)
    tensor_bases = torch.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], dim=1)
    return invariants, tensor_bases

class Shakeout(nn.Module):
    """
    Shakeout regularisation (generalised dropout) that replaces dropped units with
    ±alpha noise instead of zeros. See: Zhang & Xie, "Shakeout: A New Regularized
    Deep Neural Network Training Scheme".
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.0):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("Shakeout drop probability p must be in [0, 1].")
        self.p = p
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input
        device = input.device
        keep_prob = 1.0 - self.p
        keep_mask = torch.empty_like(input, device=device).bernoulli_(keep_prob)
        drop_mask = 1.0 - keep_mask
        if self.alpha == 0.0:
            noise = torch.zeros_like(input, device=device)
        else:
            noise = torch.empty_like(input, device=device).bernoulli_(0.5).mul_(2.0).sub_(1.0) * self.alpha
        out = keep_mask * input + drop_mask * noise
        if keep_prob > 0.0:
            out = out / keep_prob
        return out

def ema(arr, alpha=0.15):
    arr = np.asarray(arr, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1-alpha) * out[i-1]
    return out

def eig_desc(tensor_batch: np.ndarray):
    """Return eigen-pairs in descending λ₁ ≥ λ₂ ≥ λ₃ order."""
    vals, vecs = np.linalg.eigh(tensor_batch)              # ascending
    idx = np.argsort(-vals, axis=1)                        # descending
    rows = np.arange(vals.shape[0])[:, None]
    vals = vals[rows, idx]
    vecs = vecs[rows, :, idx]
    return vals, vecs                                     # (N,3), (N,3,3)

# [VF-GLB-01] Correct Vieillefosse curve for invariants (q,r) of A (no extra √3)
def vieillefosse_curve(n_pts: int = 600, qmin: float = -1.0, qmax: float = 0.0):
    """
    Returns (qv, +rv, -rv) for the Vieillefosse tail in the (q,r) invariants:
        (27/4) r^2 + q^3 = 0  ⇒  r = ± (2/3) (-q)^{3/2},  q ≤ 0.
    q is on the Y-axis, r on the X-axis in your figures.
    """
    q_hi = min(0.0, float(qmax))
    q_lo = float(min(qmin, q_hi))
    qv = np.linspace(q_lo, q_hi, n_pts)
    neg_q = np.maximum(-qv, 0.0)
    rv = (2.0 / 3.0) * neg_q * np.sqrt(neg_q)
    return qv, +rv, -rv

def hexbin_with_mean(ax, x, y, c=None, gridsize=200, cmap="jet", vmin=0.0, vmax=2.0):
    hex_kwargs = dict(
        gridsize=gridsize,
        cmap=cmap,
        linewidths=0.1,
        vmin=vmin,
        vmax=vmax,
        mincnt=1
    )
    if c is None:
        return ax.hexbin(x, y, **hex_kwargs)
    return ax.hexbin(x, y, C=c, reduce_C_function=np.mean, **hex_kwargs)

# ==============================================================================
# 2. DATASET CLASS FOR .MAT FILES
# ==============================================================================

class MatlabDataset(torch.utils.data.Dataset):
    def __init__(self, vel_grad_path, pressure_hessian_path, build_cache: bool = True):
        print(f"Loading data from {vel_grad_path} and {pressure_hessian_path}...")
        
        def load_mat_data(path):
            mat = scipy.io.loadmat(path)
            key = next(k for k in mat if k not in ('__header__', '__version__', '__globals__'))
            return mat[key].astype(np.float32)

        vel_grad_data = load_mat_data(vel_grad_path)
        if vel_grad_data.shape[0] == 9:
            vel_grad_data = vel_grad_data.T
        self.A = torch.from_numpy(vel_grad_data).view(-1, 3, 3).to(torch.float64)

        ph_data = load_mat_data(pressure_hessian_path)
        if ph_data.shape[0] == 9: ph_data = ph_data.T
        raw_P = torch.from_numpy(ph_data).view(-1, 3, 3).to(torch.float64)

        trace_P = torch.einsum('bii->b', raw_P).unsqueeze(-1).unsqueeze(-1)
        eye3 = torch.eye(3, dtype=raw_P.dtype, device=raw_P.device)
        self.Q = raw_P - (trace_P / 3.0) * eye3.unsqueeze(0)

        assert self.A.shape[0] == self.Q.shape[0], "Data sample counts do not match."
        self.num_samples = self.A.shape[0]
        print(f"Data loaded successfully. Found {self.num_samples} samples.")
        self.cache = {}
        if build_cache:
            self._build_feature_cache()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        if not self.cache:
            return self.A[idx], self.Q[idx]
        sample = {
            'A': self.A[idx],
            'Q': self.Q[idx],
            'epsilon': self.cache['epsilon'][idx],
            's_norm': self.cache['s_norm'][idx],
            'w': self.cache['w'][idx],
            'invariants': self.cache['invariants'][idx],
            'tensor_bases': self.cache['tensor_bases'][idx],
            'q': self.cache['q'][idx],
            'r': self.cache['r'][idx],
            'Q_prime': self.cache['Q_prime'][idx],
            'Q_hat': self.cache['Q_hat'][idx],
            'psi': self.cache['psi'][idx],
            'eig_s': self.cache['eig_s'][idx],
            'eig_Q_true': self.cache['eig_Q_true'][idx]
        }
        return sample

    def get_full_dataset(self, with_cache: bool = False):
        """Returns the full tensors for A and Q (and cache if requested)."""
        if with_cache:
            return self.A, self.Q, self.cache
        return self.A, self.Q

    def _build_feature_cache(self):
        print("Precomputing tensor invariants and auxiliary quantities...")
        with torch.no_grad():
            s_raw, w_raw, epsilon, q_scalar, r_scalar = get_tensor_derivatives(self.A)
            s_norm = normalize_strain_tensor(self.A, epsilon)
            invariants, tensor_bases = compute_tbnn_invariants_and_bases(s_norm, w_raw)
            Q_prime, Q_hat_prime, psi = process_ground_truth_Q(self.Q, epsilon)
            _, eig_s_vecs = torch.linalg.eigh(s_norm)           # ascending γ,β,α
            _, eig_Q_true_vecs = torch.linalg.eigh(Q_hat_prime) # ascending γ,β,α

        self.cache = {
            'epsilon': epsilon.contiguous(),
            's_norm': s_norm.contiguous(),
            'w': w_raw.contiguous(),
            'invariants': invariants.contiguous(),
            'tensor_bases': tensor_bases.contiguous(),
            'q': q_scalar.contiguous(),
            'r': r_scalar.contiguous(),
            'Q_prime': Q_prime.contiguous(),
            'Q_hat': Q_hat_prime.contiguous(),
            'psi': psi.contiguous(),
            'eig_s': eig_s_vecs.contiguous(),
            'eig_Q_true': eig_Q_true_vecs.contiguous()
        }
        print("Precomputation complete. Cached keys:", ", ".join(sorted(self.cache.keys())))

# ==============================================================================
# 3. MODEL DEFINITIONS (With Hard Symmetry Constraint)
# ==============================================================================

class TBNN_Q_direction(nn.Module):
    def __init__(self, hidden_layers=[50, 100, 100, 100, 50], dropout_p=0.1,
                 dropout_type: str = 'dropout', shakeout_alpha: float = 0.0):
        super().__init__()
        input_dim = 5
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for hidden_dim in hidden_layers:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim, eps=1e-8, momentum=0.5))
            input_dim = hidden_dim
        self.output_layer = nn.Linear(input_dim, 10)
        self.activation = nn.LeakyReLU(0.1)
        drop_kind = (dropout_type or 'dropout').strip().lower()
        if dropout_p <= 0.0:
            self.dropout = nn.Identity()
        elif drop_kind == 'shakeout':
            self.dropout = Shakeout(dropout_p, shakeout_alpha)
        else:
            self.dropout = nn.Dropout(dropout_p)
        self._init_weights()

    def forward(self, s, w, invariants=None, tensor_bases=None, return_penalty: bool = False):
        if invariants is None or tensor_bases is None:
            invariants, tensor_bases = compute_tbnn_invariants_and_bases(s, w)
        x = invariants
        for linear, bn in zip(self.linears, self.bns):
            x = linear(x)
            x = self.activation(x)
            x = bn(x)
            x = self.dropout(x)
        g = F.relu(self.output_layer(x))
        Q_hat_raw = torch.einsum('bn,bnij->bij', g, tensor_bases)

        sym_penalty = None
        if return_penalty:
            antisym = 0.5 * (Q_hat_raw - Q_hat_raw.transpose(-2, -1))
            sym_penalty = torch.mean(torch.sum(antisym * antisym, dim=(1, 2)))

        Q_hat_sym = 0.5 * (Q_hat_raw + Q_hat_raw.transpose(-2, -1))
        eye = torch.eye(3, device=Q_hat_sym.device, dtype=Q_hat_sym.dtype).unsqueeze(0)
        trace = torch.diagonal(Q_hat_sym, dim1=-2, dim2=-1).sum(-1, keepdim=True) / 3.0
        Q_hat_traceless = Q_hat_sym - trace.unsqueeze(-1) * eye
        norm = torch.sqrt(torch.sum(Q_hat_traceless * Q_hat_traceless, dim=(1, 2), keepdim=True) + 1e-12)
        Q_hat_prime_pred = Q_hat_traceless / norm
        if return_penalty:
            return Q_hat_prime_pred, sym_penalty
        return Q_hat_prime_pred

    def _init_weights(self):
        for idx, linear in enumerate(self.linears):
            nn.init.kaiming_uniform_(linear.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

class FCNN_psi_magnitude(nn.Module):
    def __init__(self, hidden_layers=[50, 80, 50], dropout_p=0.1,
                 dropout_type: str = 'dropout', shakeout_alpha: float = 0.0):
        super().__init__()
        input_dim = 2
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for hidden_dim in hidden_layers:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim, eps=1e-8, momentum=0.5))
            input_dim = hidden_dim
        self.output_layer = nn.Linear(input_dim, 1)
        self.activation = nn.LeakyReLU(0.1)
        drop_kind = (dropout_type or 'dropout').strip().lower()
        if dropout_p <= 0.0:
            self.dropout = nn.Identity()
        elif drop_kind == 'shakeout':
            self.dropout = Shakeout(dropout_p, shakeout_alpha)
        else:
            self.dropout = nn.Dropout(dropout_p)
        self._init_weights()

    def forward(self, q, r):
        inputs = torch.stack([q, r], dim=1)
        x = inputs
        for linear, bn in zip(self.linears, self.bns):
            x = linear(x)
            x = self.activation(x)
            x = bn(x)
            x = self.dropout(x)
        return F.relu(self.output_layer(x)).squeeze()

    def _init_weights(self):
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

# ==============================================================================
# 4. CUSTOM LEARNING-RATE SCHEDULER
# ==============================================================================


class ChainedCosineExpLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup → optional cosine anneal → exponential decay scheduler."""

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps: int,
        pct_start: float,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        three_phase: bool = False,
        anneal_strategy: str = 'cos',
        last_epoch: int = -1
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        pct_start = float(np.clip(pct_start, 1e-6, 0.999))
        self.total_steps = int(total_steps)
        self.three_phase = bool(three_phase)
        strategy = (anneal_strategy or 'cos').strip().lower()
        if strategy not in {'cos', 'linear'}:
            raise ValueError("anneal_strategy must be 'cos' or 'linear'.")
        self.anneal_strategy = strategy

        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("max_lr length mismatch with optimizer param groups.")
            self.max_lrs = [float(lr) for lr in max_lr]
        else:
            self.max_lrs = [float(max_lr)] * len(optimizer.param_groups)

        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.base_lrs = [lr / self.div_factor for lr in self.max_lrs]
        self.min_lrs = [lr / self.final_div_factor for lr in self.max_lrs]

        warmup_steps = max(1, int(round(self.total_steps * pct_start)))
        if warmup_steps >= self.total_steps:
            warmup_steps = max(1, self.total_steps - 1)
        self.warmup_steps = warmup_steps
        if self.three_phase:
            remaining = max(1, self.total_steps - self.warmup_steps)
            self.anneal_steps = max(1, remaining // 2)
        else:
            self.anneal_steps = 0
        self.decay_steps = max(1, self.total_steps - self.warmup_steps - self.anneal_steps)
        self.decay_start_lrs = self.base_lrs if self.three_phase else self.max_lrs

        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            group['lr'] = base_lr
            group.setdefault('initial_lr', base_lr)

        super().__init__(optimizer, last_epoch)

    def _phase_progress(self, step: int) -> tuple[str, float]:
        if step < self.warmup_steps:
            return 'warmup', (step + 1) / self.warmup_steps
        if self.three_phase:
            boundary = self.warmup_steps + self.anneal_steps
            if step < boundary:
                return 'anneal', (step - self.warmup_steps + 1) / self.anneal_steps
        decay_start = self.warmup_steps + self.anneal_steps
        return 'decay', (step - decay_start + 1) / self.decay_steps

    def _shape_factor(self, progress: float, invert: bool = False) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        if self.anneal_strategy == 'cos':
            value = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            value = progress
        return 1.0 - value if invert else value

    def get_lr(self):
        step = self.last_epoch
        if step < 0:
            return self.base_lrs

        phase, progress = self._phase_progress(step)
        progress = float(np.clip(progress, 0.0, 1.0))
        lrs = []
        for idx, (base_lr, max_lr, min_lr) in enumerate(zip(self.base_lrs, self.max_lrs, self.min_lrs)):
            if phase == 'warmup':
                factor = self._shape_factor(progress)
                lr = base_lr + factor * (max_lr - base_lr)
            elif phase == 'anneal':
                factor = self._shape_factor(progress, invert=True)
                lr = base_lr + factor * (max_lr - base_lr)
            else:
                start_lr = self.decay_start_lrs[idx]
                denom = max(start_lr, 1e-12)
                decay_ratio = max(min_lr / denom, 1e-12)
                lr = start_lr * (decay_ratio ** progress)
                lr = max(lr, min_lr)
            lrs.append(lr)
        return lrs

# 4. LOSS FUNCTIONS (With Physics Constraints and Log-Cosh)
# ==============================================================================
def euler_angle_loss(Q_hat_prime_pred, Q_hat_prime_true, s_true, precomputed=None):
    try:
        if precomputed is not None and 's_eigvecs' in precomputed:
            e_s_vecs = precomputed['s_eigvecs']
        else:
            _, e_s_vecs = torch.linalg.eigh(s_true)
        e_gamma_s, e_beta_s, e_alpha_s = e_s_vecs[:, :, 0], e_s_vecs[:, :, 1], e_s_vecs[:, :, 2]

        if precomputed is not None and 'Q_hat_true_eigvecs' in precomputed:
            e_p_vecs_true = precomputed['Q_hat_true_eigvecs']
        else:
            _, e_p_vecs_true = torch.linalg.eigh(Q_hat_prime_true)
        e_gamma_p_true, e_beta_p_true, e_alpha_p_true = e_p_vecs_true[:, :, 0], e_p_vecs_true[:, :, 1], e_p_vecs_true[:, :, 2]
        _, e_p_vecs_pred = torch.linalg.eigh(Q_hat_prime_pred)
        e_gamma_p_pred, e_beta_p_pred, e_alpha_p_pred = e_p_vecs_pred[:, :, 0], e_p_vecs_pred[:, :, 1], e_p_vecs_pred[:, :, 2]

        cos_zeta_true = torch.einsum('bi,bi->b', e_gamma_p_true, e_gamma_s)
        e_proj_prime_true = e_alpha_p_true - torch.einsum('bi,bi->b', e_alpha_p_true, e_gamma_s).unsqueeze(1) * e_gamma_s
        e_proj_true = e_proj_prime_true / (torch.norm(e_proj_prime_true, dim=1, keepdim=True) + 1e-8)
        cos_theta_true = torch.einsum('bi,bi->b', e_alpha_s, e_proj_true)
        e_norm_true = torch.cross(e_proj_true, e_gamma_s)
        cos_eta_true = torch.einsum('bi,bi->b', e_beta_p_true, e_norm_true)

        cos_zeta_pred = torch.einsum('bi,bi->b', e_gamma_p_pred, e_gamma_s)
        e_proj_prime_pred = e_alpha_p_pred - torch.einsum('bi,bi->b', e_alpha_p_pred, e_gamma_s).unsqueeze(1) * e_gamma_s
        e_proj_pred = e_proj_prime_pred / (torch.norm(e_proj_prime_pred, dim=1, keepdim=True) + 1e-8)
        cos_theta_pred = torch.einsum('bi,bi->b', e_alpha_s, e_proj_pred)
        e_norm_pred = torch.cross(e_proj_pred, e_gamma_s)
        cos_eta_pred = torch.einsum('bi,bi->b', e_beta_p_pred, e_norm_pred)
        
        L1 = torch.sum((torch.abs(cos_zeta_true) - torch.abs(cos_zeta_pred))**2) / (torch.sum(cos_zeta_true**2) + 1e-8)
        L2 = torch.sum((torch.abs(cos_theta_true) - torch.abs(cos_theta_pred))**2) / (torch.sum(cos_theta_true**2) + 1e-8)
        L3 = torch.sum((torch.abs(cos_eta_true) - torch.abs(cos_eta_pred))**2) / (torch.sum(cos_eta_true**2) + 1e-8)
        loss_J = L1 + L2 + L3
        
        components = {
            'L1': L1.item(), 'L2': L2.item(), 'L3': L3.item(),
            'cos_zeta_err': torch.mean(torch.abs(torch.abs(cos_zeta_true) - torch.abs(cos_zeta_pred))).item(),
            'cos_theta_err': torch.mean(torch.abs(torch.abs(cos_theta_true) - torch.abs(cos_theta_pred))).item(),
            'cos_eta_err': torch.mean(torch.abs(torch.abs(cos_eta_true) - torch.abs(cos_eta_pred))).item()
        }
        return loss_J, components
    except torch.linalg.LinAlgError:
        return torch.tensor(0.0, device=Q_hat_prime_pred.device, requires_grad=True), {}

# ==============================================================================
# 5. TRAINING PIPELINE (FP64, Grad Clipping)
# ==============================================================================
class AIBMTrainer:
    def __init__(self, model_Q, model_psi, train_loader, val_loader,
                epochs=400, learning_rate_Q=1e-3, learning_rate_psi=5e-3,
                grad_clip_value=5.0, symm_loss_weight=0.5,
                print_every=1, metrics_save_path=None,
                ema_smoothing=0.15, log_header_interval=100,
                dropout_p=0.1, lr_scheduler_type='linear', scheduler_warmup: float | None = 0.1,
                snapshot_dir=None, log_interval_type='epoch',
                log_interval_seconds=60.0, enable_profiler=True,
                compile_models=False,
                optimizer_name_q: str = 'adamax', optimizer_name_psi: str = 'adamax',
                weight_decay_q: float = 0.0, weight_decay_psi: float = 0.0,
                beta1_q: float = 0.9, beta2_q: float = 0.999,
                beta1_psi: float = 0.9, beta2_psi: float = 0.999,
                nadam_momentum_decay_q: float = 0.004,
                nadam_momentum_decay_psi: float = 0.004,
                chained_div_factor: float = 25.0,
                chained_final_div_factor: float = 10000.0,
                chained_three_phase: bool = False,
                chained_anneal_strategy: str = 'cos'):
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dtype = torch.float64
        self.model_Q = model_Q.to(self.device, dtype=self.train_dtype)
        self.model_psi = model_psi.to(self.device, dtype=self.train_dtype)
        self._compiled = False
        if compile_models and hasattr(torch, "compile"):
            try:
                self.model_Q = torch.compile(self.model_Q)
                self.model_psi = torch.compile(self.model_psi)
                self._compiled = True
            except Exception as compile_err:
                warnings.warn(f"torch.compile failed ({compile_err}); continuing without compilation.", stacklevel=2)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_name_Q = (optimizer_name_q or 'adamax').strip().lower()
        self.optimizer_name_psi = (optimizer_name_psi or 'adamax').strip().lower()
        if self.optimizer_name_Q not in {'adamax', 'nadam'}:
            raise ValueError("optimizer_name_q must be 'adamax' or 'nadam'.")
        if self.optimizer_name_psi not in {'adamax', 'nadam'}:
            raise ValueError("optimizer_name_psi must be 'adamax' or 'nadam'.")
        self.weight_decay_Q = float(weight_decay_q)
        self.weight_decay_psi = float(weight_decay_psi)
        self.betas_Q = (float(beta1_q), float(beta2_q))
        self.betas_psi = (float(beta1_psi), float(beta2_psi))
        self.nadam_decay_Q = float(nadam_momentum_decay_q)
        self.nadam_decay_psi = float(nadam_momentum_decay_psi)
        self.optimizer_Q = self._create_optimizer(
            self.optimizer_name_Q,
            self.model_Q.parameters(),
            learning_rate_Q,
            self.weight_decay_Q,
            self.betas_Q,
            self.nadam_decay_Q
        )
        self.optimizer_psi = self._create_optimizer(
            self.optimizer_name_psi,
            self.model_psi.parameters(),
            learning_rate_psi,
            self.weight_decay_psi,
            self.betas_psi,
            self.nadam_decay_psi
        )
        self.scheduler_type = lr_scheduler_type.strip().lower().replace(' ', '_')
        if self.scheduler_type in {'warmup_cosine_decay', 'warmup-cosine-decay'}:
            self.scheduler_type = 'warmup_cosine'
        self.scheduler_warmup_value = scheduler_warmup
        self._warmup_config = self._resolve_warmup_config(scheduler_warmup)
        self.chained_three_phase = bool(chained_three_phase)
        self.chained_div_factor = float(chained_div_factor)
        self.chained_final_div_factor = float(chained_final_div_factor)
        strategy = (chained_anneal_strategy or 'cos').strip().lower()
        if strategy not in {'cos', 'linear'}:
            raise ValueError("chained_anneal_strategy must be 'cos' or 'linear'.")
        self.chained_anneal_strategy = strategy
        self.scheduler_Q = self._build_scheduler(self.optimizer_Q)
        self.scheduler_psi = self._build_scheduler(self.optimizer_psi)
        self.grad_clip_value = grad_clip_value
        self.symm_loss_weight = float(symm_loss_weight)
        interval = (log_interval_type or 'epoch').strip().lower()
        if interval in {'epoch', 'epochs'}:
            self.log_interval_type = 'epoch'
        elif interval in {'time', 'seconds', 'wall', 'wall_time'}:
            self.log_interval_type = 'time'
        else:
            warnings.warn(f"Unknown log_interval_type '{log_interval_type}', defaulting to 'epoch'.")
            self.log_interval_type = 'epoch'
        self.print_every = max(1, int(print_every))
        self.log_interval_seconds = max(0.0, float(log_interval_seconds))
        self.metrics_save_path = metrics_save_path
        self.history = []
        self.train_metrics = []
        self.val_metrics = []
        self.best = {}
        self.snapshot_dir = str(snapshot_dir) if snapshot_dir else None
        self.dropout_p = float(dropout_p)
        self._set_model_dropout(self.dropout_p)

        # For EMA smoothing and logging
        self.ema_alpha = ema_smoothing
        self.log_header_interval = max(1, log_header_interval)
        self._log_line_count = 0
        self._last_log_time = None
        self.enable_profiler = bool(enable_profiler)

    def train(self, epochs=None):
        epochs = epochs if epochs else self.epochs
        header = (
            f"{'Epoch':>5} | {'train_euler':>11} | {'val_euler':>11} | "
            f"{'train_psi_RMSE':>16} | {'val_psi_RMSE':>16} | "
            f"{'t_epoch(s)':>12}"
        )
        print('\n' + header)
        print('-' * len(header))
        import time
        self._last_log_time = None
        if self._compiled:
            print("[info] torch.compile enabled for model_Q/model_psi")

        for epoch in range(epochs):
            epoch_idx = epoch + 1

            # ---------- Training ----------
            self.model_Q.train()
            self.model_psi.train()
            train_start = time.time()
            train_stats = self._epoch_pass(self.train_loader, mode='train')
            train_time = time.time() - train_start
            current_lr_Q = self.optimizer_Q.param_groups[0]['lr']
            current_lr_psi = self.optimizer_psi.param_groups[0]['lr']
            train_stats = {
                **train_stats,
                'lr_Q': current_lr_Q,
                'lr_psi': current_lr_psi,
                'time': train_time
            }

            # ---------- Validation ----------
            self.model_Q.eval()
            self.model_psi.eval()
            with torch.no_grad():
                val_start = time.time()
                val_stats = self._epoch_pass(self.val_loader, mode='val')
                val_time = time.time() - val_start
            val_stats = {**val_stats, 'time': val_time}

            # ---------- Scheduler (per-epoch) ----------
            if self.scheduler_Q:
                self.scheduler_Q.step()
            if self.scheduler_psi:
                self.scheduler_psi.step()

            # ---------- Record ----------
            flat_record = {'epoch': epoch_idx}
            flat_record.update({f"train_{k}": v for k, v in train_stats.items()})
            flat_record.update({f"val_{k}": v for k, v in val_stats.items()})
            self.history.append(flat_record)
            self.train_metrics.append({'epoch': epoch_idx, **train_stats})
            self.val_metrics.append({'epoch': epoch_idx, **val_stats})

            total_time = train_stats['time'] + val_stats['time']
            now = time.time()
            train_psi_rmse = train_stats.get('psi_rmse', float('nan'))
            val_psi_rmse = val_stats.get('psi_rmse', float('nan'))

            if self._should_log(epoch_idx, now):
                if self._log_line_count % self.log_header_interval == 0 and self._log_line_count != 0:
                    print('\n' + header)
                    print('-' * len(header))
                print(
                    f"{epoch_idx:5d} | "
                    f"{self._format_sci(train_stats['euler']):>11} | "
                    f"{self._format_sci(val_stats['euler']):>11} | "
                    f"{self._format_sci(train_psi_rmse):>16} | "
                    f"{self._format_sci(val_psi_rmse):>16} | "
                    f"{self._format_sci(total_time):>12}"
                )
                self._log_line_count += 1
                self._last_log_time = now

            # ---------- Best tracking / snapshots ----------
            best_val_euler = self.best.get('euler', float('inf'))
            current_val_euler = val_stats['euler']
            if not math.isnan(current_val_euler) and current_val_euler < best_val_euler:
                self.best = {'epoch': epoch_idx, **val_stats}
                if self.snapshot_dir:
                    self.save_snapshot(epoch_idx, val_stats, save_dir=self.snapshot_dir, tag='best')
            # Save rolling last snapshot
            if self.snapshot_dir:
                self.save_snapshot(epoch_idx, val_stats, save_dir=self.snapshot_dir, tag='last')

        if self.best:
            print("\nBest validation metrics (by Euler loss):")
            print(self.best)

        if self.enable_profiler:
            train_profiles = [m.get('profile_total') for m in self.train_metrics if isinstance(m.get('profile_total'), (int, float))]
            val_profiles = [m.get('profile_total') for m in self.val_metrics if isinstance(m.get('profile_total'), (int, float))]
            if train_profiles or val_profiles:
                def _summary(arr):
                    arr = [x for x in arr if x is not None and not math.isnan(x)]
                    return (np.mean(arr), np.median(arr)) if arr else (float('nan'), float('nan'))
                train_mean, train_med = _summary(train_profiles)
                val_mean, val_med = _summary(val_profiles)
                print("\nProfiler summary (seconds per epoch):")
                print({
                    'train_mean': self._format_sci(train_mean),
                    'train_median': self._format_sci(train_med),
                    'val_mean': self._format_sci(val_mean),
                    'val_median': self._format_sci(val_med)
                })

        # Save to DataFrame for later analysis
        self.history_df = pd.DataFrame(self.history)
        if self.metrics_save_path:
            self.history_df.to_csv(self.metrics_save_path, index=False)

    @staticmethod
    def _format_sci(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "nan"
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.2e}"

    def _should_log(self, epoch_idx: int, current_time: float) -> bool:
        if epoch_idx == 1:
            return True
        if self.log_interval_type == 'epoch':
            return (epoch_idx % self.print_every) == 0
        if self.log_interval_seconds <= 0.0:
            return True
        if self._last_log_time is None:
            return True
        return (current_time - self._last_log_time) >= self.log_interval_seconds

    def _set_model_dropout(self, p: float):
        p = float(np.clip(p, 0.0, 1.0))
        for module in self.model_Q.modules():
            if isinstance(module, (nn.Dropout, Shakeout)):
                module.p = p
        for module in self.model_psi.modules():
            if isinstance(module, (nn.Dropout, Shakeout)):
                module.p = p
        self.dropout_p = p

    def _resolve_warmup_config(self, warmup_value):
        if warmup_value is None:
            return {'input': None, 'pct': 0.0, 'epochs': 0}
        try:
            value = float(warmup_value)
        except (TypeError, ValueError):
            return {'input': warmup_value, 'pct': 0.0, 'epochs': 0}
        if value <= 0.0:
            return {'input': value, 'pct': 0.0, 'epochs': 0}
        if value <= 1.0:
            pct = float(np.clip(value, 1e-6, 0.999))
            epochs = max(1, int(round(self.epochs * pct)))
        else:
            epochs = int(round(value))
            epochs = int(np.clip(epochs, 1, max(1, self.epochs)))
            pct = epochs / max(1, self.epochs)
        return {'input': value, 'pct': pct, 'epochs': epochs}

    @staticmethod
    def _create_optimizer(name: str, params, lr: float, weight_decay: float,
                          betas: tuple[float, float], nadam_decay: float):
        name = (name or 'adamax').strip().lower()
        if name == 'adamax':
            return torch.optim.Adamax(params, lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay)
        if name == 'nadam':
            return torch.optim.NAdam(
                params,
                lr=lr,
                betas=betas,
                eps=1e-8,
                weight_decay=weight_decay,
                momentum_decay=nadam_decay,
                decoupled_weight_decay=True
            )
        raise ValueError("Unsupported optimizer. Choose 'adamax' or 'nadam'.")

    def _build_scheduler(self, optimizer):
        sched = self.scheduler_type
        if sched == 'none':
            return None
        if sched == 'linear':
            def lr_lambda(epoch):
                if self.epochs <= 1:
                    return 1.0
                return max(0.0, 1.0 - (epoch / (self.epochs - 1)))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if sched == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0.0)
        if sched in {'warmup_cosine', 'warmup-cosine', 'warmup_cosine_decay', 'warmupcosine'}:
            warmup = min(self._warmup_config['epochs'], max(0, self.epochs - 1))
            def lr_lambda(epoch):
                if warmup > 0 and epoch < warmup:
                    return float(epoch + 1) / float(warmup)
                if self.epochs <= warmup + 1:
                    return 1.0
                progress = (epoch - warmup) / max(1, (self.epochs - warmup - 1))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if sched in {'chained', 'chained_cosine', 'chained_cosine_exp'}:
            max_lr = [group['lr'] for group in optimizer.param_groups]
            total_steps = max(1, self.epochs)
            return ChainedCosineExpLR(
                optimizer,
                max_lr=max_lr if len(max_lr) > 1 else max_lr[0],
                total_steps=total_steps,
                pct_start=self._warmup_config['pct'],
                div_factor=self.chained_div_factor,
                final_div_factor=self.chained_final_div_factor,
                three_phase=self.chained_three_phase,
                anneal_strategy=self.chained_anneal_strategy
            )
        raise ValueError(f"Unknown scheduler type '{self.scheduler_type}'. Choose from 'none', 'linear', 'cosine', 'warmup_cosine', 'chained'.")

    def _prepare_raw_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            A_batch, Q_batch = batch
        else:
            raise TypeError("Expected (A_batch, Q_batch) tuple when cache is unavailable.")
        A_batch = A_batch.to(dtype=self.train_dtype)
        Q_batch = Q_batch.to(dtype=self.train_dtype)
        with torch.no_grad():
            s_raw, w_raw, epsilon, q_vals, r_vals = get_tensor_derivatives(A_batch)
            s_norm = normalize_strain_tensor(A_batch, epsilon)
            invariants, tensor_bases = compute_tbnn_invariants_and_bases(s_norm, w_raw)
            Q_prime, Q_hat_prime, psi = process_ground_truth_Q(Q_batch, epsilon)
            _, eig_s_vecs = torch.linalg.eigh(s_norm)
            _, eig_Q_true_vecs = torch.linalg.eigh(Q_hat_prime)
        return {
            's_norm': s_norm,
            'w': w_raw,
            'invariants': invariants,
            'tensor_bases': tensor_bases,
            'q': q_vals,
            'r': r_vals,
            'Q_prime': Q_prime,
            'Q_hat': Q_hat_prime,
            'psi': psi,
            'eig_s': eig_s_vecs,
            'eig_Q_true': eig_Q_true_vecs
        }

    def _epoch_pass(self, loader, mode='train'):
        psi_rmses = []
        psi_r2s = []
        euler_losses, euler_L1, euler_L2, euler_L3 = [], [], [], []
        symm_losses = []
        profiler = collections.defaultdict(float) if self.enable_profiler else None
        import time

        for batch in loader:
            sample = batch if isinstance(batch, dict) else self._prepare_raw_batch(batch)

            t_device = time.perf_counter()
            s_norm = sample['s_norm'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            w = sample['w'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            invariants = sample['invariants'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            tensor_bases = sample['tensor_bases'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            q_vals = sample['q'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            r_vals = sample['r'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            Q_hat_t = sample['Q_hat'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            psi_t = sample['psi'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            eig_s = sample['eig_s'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            eig_Q_true = sample['eig_Q_true'].to(self.device, dtype=self.train_dtype, non_blocking=True)
            if profiler is not None:
                profiler['to_device'] += time.perf_counter() - t_device
            t_forward = time.perf_counter()
            Q_hat_p, sym_penalty = self.model_Q(s_norm, w, invariants=invariants, tensor_bases=tensor_bases, return_penalty=True)
            psi_p = self.model_psi(q_vals, r_vals)
            if profiler is not None:
                profiler['forward'] += time.perf_counter() - t_forward

            t_loss = time.perf_counter()
            q_mse = F.mse_loss(Q_hat_p, Q_hat_t)
            psi_mse = F.mse_loss(psi_p, psi_t)
            q_rmse = torch.sqrt(q_mse + 1e-12)
            psi_rmse = torch.sqrt(psi_mse + 1e-12)

            precomp = {'s_eigvecs': eig_s, 'Q_hat_true_eigvecs': eig_Q_true}
            loss_euler, euler_dict = euler_angle_loss(Q_hat_p, Q_hat_t, s_norm, precomputed=precomp)

            finite_components = torch.stack([q_rmse, psi_rmse, loss_euler, sym_penalty])
            if not torch.isfinite(finite_components).all():
                warnings.warn("Non-finite loss components encountered; skipping batch.", stacklevel=2)
                continue

            # q_scale = torch.clamp(torch.std(Q_hat_t.detach()), min=1e-6)
            # psi_scale = torch.clamp(torch.std(psi_t.detach()), min=1e-6)
            # euler_scale = torch.clamp(torch.std(torch.tensor([1.0, 1.0, 1.0], device=loss_euler.device, dtype=loss_euler.dtype)), min=1e-6)
            total_loss = q_rmse + psi_rmse + loss_euler + (self.symm_loss_weight * sym_penalty)
            if profiler is not None:
                profiler['loss'] += time.perf_counter() - t_loss

            psi_rmses.append(float(psi_rmse.detach()))
            r2_val = self._r2(psi_p, psi_t)
            if math.isfinite(r2_val):
                psi_r2s.append(r2_val)

            symm_losses.append(float(sym_penalty.detach()))
            euler_losses.append(float(loss_euler.detach()))
            euler_L1.append(euler_dict.get('L1', 0.0))
            euler_L2.append(euler_dict.get('L2', 0.0))
            euler_L3.append(euler_dict.get('L3', 0.0))

            if mode == 'train':
                self.optimizer_Q.zero_grad(set_to_none=True)
                self.optimizer_psi.zero_grad(set_to_none=True)

                t_backward = time.perf_counter()
                total_loss.backward()
                if profiler is not None:
                    profiler['backward'] += time.perf_counter() - t_backward

                t_opt = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(self.model_Q.parameters(), self.grad_clip_value)
                torch.nn.utils.clip_grad_norm_(self.model_psi.parameters(), self.grad_clip_value)
                self.optimizer_Q.step()
                self.optimizer_psi.step()
                if profiler is not None:
                    profiler['optim'] += time.perf_counter() - t_opt

        def safe_mean(values):
            if not values:
                return float('nan')
            arr = np.asarray(values, dtype=np.float64)
            mask = np.isfinite(arr)
            if not mask.any():
                return float('nan')
            return float(np.mean(arr[mask]))

        res = {
            'euler': safe_mean(euler_losses),
            'L1': safe_mean(euler_L1),
            'L2': safe_mean(euler_L2),
            'L3': safe_mean(euler_L3),
            'symm_loss': safe_mean(symm_losses),
            'psi_rmse': safe_mean(psi_rmses),
            'psi_r2': safe_mean(psi_r2s),
        }
        if profiler is not None:
            for key, value in profiler.items():
                res[f'profile_{key}'] = value
            res['profile_total'] = sum(profiler.values())
        return res

    @staticmethod
    def _r2(pred, true):
        pred_np = pred.detach().cpu().numpy().reshape(-1)
        true_np = true.detach().cpu().numpy().reshape(-1)
        mask = np.isfinite(pred_np) & np.isfinite(true_np)
        if not np.any(mask):
            return float('nan')
        pred_np = pred_np[mask]
        true_np = true_np[mask]
        mean_true = np.mean(true_np)
        ss_res = np.sum((true_np - pred_np) ** 2)
        ss_tot = np.sum((true_np - mean_true) ** 2) + 1e-8
        return 1 - ss_res / ss_tot

    def plot_history(self, save_dir=None):
        if not self.train_metrics or not self.val_metrics:
            return
        save_dir = Path(save_dir) if save_dir else None
        train_df = pd.DataFrame(self.train_metrics)
        val_df = pd.DataFrame(self.val_metrics)
        self._plot_phase_history(train_df, phase='train', save_dir=save_dir)
        self._plot_phase_history(val_df, phase='val', save_dir=save_dir)

    def _plot_phase_history(self, df: pd.DataFrame, phase: str, save_dir: Path | None):
        epochs = df['epoch'].to_numpy()
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Euler components
        axs[0, 0].plot(epochs, df['euler'], 'C0-', lw=2, label='Euler(Q)')
        axs[0, 0].plot(epochs, ema(df['euler'], self.ema_alpha), 'C0--', lw=1.5, label='Euler EMA')
        for comp, color in zip(['L1', 'L2', 'L3'], ['C2', 'C3', 'C4']):
            if comp in df.columns:
                axs[0, 0].plot(epochs, df[comp], color+'-', lw=1.2, alpha=0.7, label=comp)
        axs[0, 0].set_ylabel('Euler loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].grid(True, ls=':', alpha=0.6)
        axs[0, 0].legend()

        # Panel 2: ψ metrics
        axs[0, 1].plot(epochs, df['psi_rmse'], 'C1-', lw=2, label='ψ RMSE')
        axs[0, 1].plot(epochs, ema(df['psi_rmse'], self.ema_alpha), 'C1--', lw=1.5, label='ψ RMSE EMA')
        if 'psi_logcosh' in df.columns:
            axs[0, 1].plot(epochs, df['psi_logcosh'], 'C6-', lw=1.2, alpha=0.8, label='ψ Log-Cosh')
        if 'psi_mse' in df.columns:
            axs[0, 1].plot(epochs, df['psi_mse'], 'C7-', lw=1.2, alpha=0.8, label='ψ MSE')
        if 'psi_r2' in df.columns:
            axs[0, 1].plot(epochs, df['psi_r2'], 'C8-', lw=1.2, alpha=0.8, label='ψ R²')
        axs[0, 1].set_ylabel('ψ metrics')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].grid(True, ls=':', alpha=0.6)
        axs[0, 1].legend()

        # Panel 3: Q metrics (optional)
        q_panel_plotted = False
        if 'Q_rmse' in df.columns:
            axs[1, 0].plot(epochs, df['Q_rmse'], 'C5-', lw=2, label='Q RMSE')
            axs[1, 0].plot(epochs, ema(df['Q_rmse'], self.ema_alpha), 'C5--', lw=1.5, label='Q RMSE EMA')
            q_panel_plotted = True
        if 'tensor_mse' in df.columns:
            axs[1, 0].plot(epochs, df['tensor_mse'], 'C9-', lw=1.2, alpha=0.8, label='Q MSE')
            q_panel_plotted = True
        if 'tensor_logcosh' in df.columns:
            axs[1, 0].plot(epochs, df['tensor_logcosh'], 'C10-', lw=1.2, alpha=0.8, label='Q Log-Cosh')
            q_panel_plotted = True
        if 'hess_rmse' in df.columns:
            axs[1, 0].plot(epochs, df['hess_rmse'], 'C11-', lw=1.2, alpha=0.8, label='Max Hessian eig.')
            q_panel_plotted = True
        if 'symm_loss' in df.columns:
            axs[1, 0].plot(epochs, df['symm_loss'], 'C3-', lw=1.2, alpha=0.8, label='Symmetry loss')
            q_panel_plotted = True
        if q_panel_plotted:
            axs[1, 0].set_ylabel('Q metrics')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].grid(True, ls=':', alpha=0.6)
            axs[1, 0].legend()
        else:
            axs[1, 0].axis('off')

        # Panel 4: Learning rate curves (if available)
        if ('lr_Q' in df.columns) or ('lr_psi' in df.columns):
            if 'lr_Q' in df.columns:
                axs[1, 1].plot(epochs, df['lr_Q'], 'C11-', lw=1.5, label='LR_Q')
            if 'lr_psi' in df.columns:
                axs[1, 1].plot(epochs, df['lr_psi'], 'C12-', lw=1.5, label='LR_ψ')
            axs[1, 1].set_ylabel('Learning rate')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_yscale('log')
            axs[1, 1].grid(True, ls=':', alpha=0.6)
            axs[1, 1].legend()
        else:
            axs[1, 1].axis('off')

        fig.suptitle(f"AIBM {phase.title()} Metrics", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir:
            out_path = save_dir / f"{phase}_metrics.png"
            fig.savefig(out_path, dpi=300)
        plt.close(fig)

    def save_snapshot(self, epoch, val_metrics, save_dir, tag=None):
        os.makedirs(save_dir, exist_ok=True)
        state = {
            'epoch': epoch,
            'model_Q': self.model_Q.state_dict(),
            'model_psi': self.model_psi.state_dict(),
            'optimizer_Q': self.optimizer_Q.state_dict(),
            'optimizer_psi': self.optimizer_psi.state_dict(),
            'val_metrics': val_metrics
        }
        fname = f"snapshot_epoch{epoch:03d}"
        if tag:
            fname += f"_{tag}"
        fname += ".pt"
        torch.save(state, os.path.join(save_dir, fname))

    def load_snapshot(self, snapshot_path, strict=True):
        state = torch.load(snapshot_path, map_location=self.device, weights_only=False)
        self.model_Q.load_state_dict(state['model_Q'], strict=strict)
        self.model_psi.load_state_dict(state['model_psi'], strict=strict)
        self.optimizer_Q.load_state_dict(state['optimizer_Q'])
        self.optimizer_psi.load_state_dict(state['optimizer_psi'])
        return state


def run_bayesian_optimization(train_loader, val_loader, base_args, trials=10, epochs_per_trial=20):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for Bayesian optimisation but is not installed. "
            "Install it with `pip install optuna`."
        ) from exc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(trial: "optuna.trial.Trial") -> float:
        lr_q = trial.suggest_float('learning_rate_Q', 1e-4, 1e-2, log=True)
        lr_psi = trial.suggest_float('learning_rate_psi', 1e-4, 5e-2, log=True)
        grad_clip = trial.suggest_float('grad_clip', 1.0, 15.0)
        dropout = trial.suggest_float('dropout', 0.05, 0.35)
        lr_sched = trial.suggest_categorical('lr_scheduler', ['linear', 'cosine', 'warmup_cosine', 'none'])
        warmup = trial.suggest_float('scheduler_warmup', 0.0, 0.5)

        drop_kind = getattr(base_args, 'dropout_type', 'dropout')
        shake_alpha = getattr(base_args, 'shakeout_alpha', 0.0)
        model_Q = TBNN_Q_direction(dropout_p=dropout, dropout_type=drop_kind, shakeout_alpha=shake_alpha).to(device, dtype=torch.float32)
        model_psi = FCNN_psi_magnitude(dropout_p=dropout, dropout_type=drop_kind, shakeout_alpha=shake_alpha).to(device, dtype=torch.float32)

        trainer = AIBMTrainer(
            model_Q=model_Q,
            model_psi=model_psi,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs_per_trial,
            learning_rate_Q=lr_q,
            learning_rate_psi=lr_psi,
            grad_clip_value=grad_clip,
            symm_loss_weight=0.0,
            print_every=max(1, epochs_per_trial // 5),
            metrics_save_path=None,
            ema_smoothing=base_args.ema_smoothing,
            log_header_interval=max(1, base_args.log_header_interval),
            dropout_p=dropout,
            lr_scheduler_type=lr_sched,
            scheduler_warmup=warmup,
            snapshot_dir=None,
            log_interval_type=base_args.log_interval_type,
            log_interval_seconds=base_args.log_interval_seconds,
            enable_profiler=not getattr(base_args, 'disable_profiler', False),
            compile_models=getattr(base_args, 'compile_models', False),
            optimizer_name_q=getattr(base_args, 'optimizer_q', 'adamax'),
            optimizer_name_psi=getattr(base_args, 'optimizer_psi', 'adamax'),
            weight_decay_q=getattr(base_args, 'weight_decay_q', 0.0),
            weight_decay_psi=getattr(base_args, 'weight_decay_psi', 0.0),
            beta1_q=getattr(base_args, 'beta1_q', 0.9),
            beta2_q=getattr(base_args, 'beta2_q', 0.999),
            beta1_psi=getattr(base_args, 'beta1_psi', 0.9),
            beta2_psi=getattr(base_args, 'beta2_psi', 0.999),
            nadam_momentum_decay_q=getattr(base_args, 'nadam_momentum_decay_q', 0.004),
            nadam_momentum_decay_psi=getattr(base_args, 'nadam_momentum_decay_psi', 0.004),
            chained_div_factor=getattr(base_args, 'chained_div_factor', 25.0),
            chained_final_div_factor=getattr(base_args, 'chained_final_div_factor', 10000.0),
            chained_three_phase=getattr(base_args, 'chained_three_phase', False),
            chained_anneal_strategy=getattr(base_args, 'chained_anneal_strategy', 'cos')
        )

        trainer.train(epochs=epochs_per_trial)
        return trainer.best.get('euler', float('inf'))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_trial



# ==============================================================================
# 6. Plotting and Visualizations class, as per paper
# ==============================================================================
# ──────────────────────────────────────────────────────────────────────────────
#  AIBM Visualizer – clean-slate rewrite
#  ─────────────────────────────────────────────────────────────────────────────
#  * Correct eigenvector ordering  (descending λ)
#  * Consistent normalisation      (shared σ for φ, common ψ range)
#  * Symmetric DNS / model plots   (every panel shows both curves)
#  * Robust defaults               (KDE bandwidth, clipping, resolution)
# ──────────────────────────────────────────────────────────────────────────────

class AIBMVisualizer:
    """
    Visual diagnostics for AIBM with DNS baselines and DNS-vs-Model overlays.

    Conventions
    -----------
    • (x, y) = (r, q) on all (q,r) plots
    • Vieillefosse tail (correct scaling):
          27 r^2 + 4 q^3 = 0  ⇒  r = ± (2/3) (-q)^{3/2},  q ≤ 0
    • Eigen-order used for labeling is ASCENDING (γ, β, α) = (λ_min, λ_mid, λ_max)
      to match the paper's notation and the training loss utilities.
    • s–Q′ grid panel order (exact):
        Row 1: γ_s·γ_p, γ_s·β_p, γ_s·α_p
        Row 2: β_s·γ_p, β_s·β_p, β_s·α_p
        Row 3: α_s·γ_p, α_s·β_p, α_s·α_p
      Vorticity vs Q′ (left→right): γ_p·ω, β_p·ω, α_p·ω
    • Figures saved at 600 DPI with 01–07 prefixes.
    """

    DPI = 600

    # ------------------------------------------------------------------ #
    # Init / prepare
    # ------------------------------------------------------------------ #
    def __init__(self, model_Q, model_psi, dataset,
                 save_dir: str | Path | None = None,
                 max_samples: int | None = 300_000,
                 seed: int = 42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mQ = model_Q.to(self.device, dtype=torch.float64).eval()
        self.mP = model_psi.to(self.device, dtype=torch.float64).eval()
        self.dataset = dataset
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)

        if save_dir is None:
            save_dir = str(globals().get('RUN_DIR', Path('figs') / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._prepared = False
    
    def debug_phi_window(self):
        X = self.X_qe2_np
        Y = self.Y_true_np
        frac_in_x = np.mean((X >= -1.0) & (X <= 1.0))
        phi_center = np.mean(Y[(X > -0.2) & (X < 0.2)]) * (np.std(X, ddof=1) / (np.std(Y, ddof=1) + 1e-15))
        print({
            "X_percentiles": np.percentile(X, [0.1, 1, 5, 50, 95, 99, 99.9]).tolist(),
            "phi_center_est": float(phi_center),
            "frac_x_in_[-1,1]": float(frac_in_x)
        })

    def _prepare(self):
        if self._prepared:
            return

        A, Q = self.dataset.get_full_dataset()
        if self.max_samples is not None and len(A) > self.max_samples:
            idx = torch.from_numpy(self.rng.choice(len(A), size=self.max_samples, replace=False))
            A, Q = A[idx], Q[idx]

        A = A.to(self.device, dtype=torch.float64)
        Q = Q.to(self.device, dtype=torch.float64)

        with torch.no_grad():
            s, w, eps, q, r = get_tensor_derivatives(A)
            a = A / eps.view(-1, 1, 1)
            s_norm = 0.5 * (a + a.transpose(1, 2))

            # DNS processing
            Qp_true, Qhat_true, psi_true = process_ground_truth_Q(Q, eps)

            # Model predictions
            Qhat_pred = self.mQ(s_norm, w)
            psi_pred = self.mP(q, r)

        self.A, self.Q = A, Q
        self.s_norm, self.w, self.eps, self.q, self.r = s_norm, w, eps, q, r
        self.Qp_true, self.Qhat_true, self.psi_true = Qp_true, Qhat_true, psi_true
        self.Qhat_pred, self.psi_pred = Qhat_pred, psi_pred

        to_np = lambda t: t.detach().cpu().numpy().astype(np.float64)
        self.q_np, self.r_np, self.eps_np = to_np(q), to_np(r), to_np(eps)
        self.psi_true_np, self.psi_pred_np = to_np(psi_true), to_np(psi_pred)
        self.Qhat_true_np, self.Qhat_pred_np = to_np(Qhat_true), to_np(Qhat_pred)

        # Dimensional Q (for rotation + φ)
        self.Q_true_np = self.Qhat_true_np * self.psi_true_np[:, None, None] * (self.eps_np[:, None, None] ** 2)
        self.Q_pred_np = self.Qhat_pred_np * self.psi_pred_np[:, None, None] * (self.eps_np[:, None, None] ** 2)

        # φ vs qε²
        self.X_qe2_np = self.q_np * (self.eps_np ** 2)
        self.Y_true_np = (self.psi_true_np ** 2) * (self.eps_np ** 4)
        self.Y_pred_np = (self.psi_pred_np ** 2) * (self.eps_np ** 4)

        self._prepared = True

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _eig_asc(M: np.ndarray):
        """Eigenpairs in ASCENDING order (γ, β, α) to match paper notation."""
        vals, vecs = np.linalg.eigh(M)  # ascending already
        return vals, vecs  # vecs[..., i] is eigenvector i (γ=0, β=1, α=2)

    @staticmethod
    def _unit_rows(X: np.ndarray, eps: float = 1e-12):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (n + eps)

    @staticmethod
    def _omega_from_w(w_np: np.ndarray):
        # ω = [w32 - w23, w13 - w31, w21 - w12]
        wx = w_np[:, 2, 1] - w_np[:, 1, 2]
        wy = w_np[:, 0, 2] - w_np[:, 2, 0]
        wz = w_np[:, 1, 0] - w_np[:, 0, 1]
        return AIBMVisualizer._unit_rows(np.stack([wx, wy, wz], axis=1))

    @staticmethod
    def _rot_z(theta_deg: float):
        t = np.deg2rad(theta_deg)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def _rotate_batch(M: np.ndarray, R: np.ndarray):
        Rt = R.T
        return np.einsum('ij,njk,kl->nil', R, M, Rt, optimize=True)

    @staticmethod
    def _vieillefosse(qmin: float, qmax: float, n: int = 800):
        """
        Vieillefosse tail in (q,r) invariants:
            27 r^2 + 4 q^3 = 0  (q ≤ 0)  ⇒  r = ± sqrt( -(4/27) q^3 )
        Returns (q_grid<=0, +r(q), -r(q)) for plotting with x=r, y=q.
        Uses sqrt form to avoid (-q)^{3/2} fp traps near q≈0⁻.
        """
        q_hi = min(0.0, float(qmax))
        q_lo = float(min(qmin, q_hi))
        qv = np.linspace(q_lo, q_hi, n)
        rv = np.sqrt(np.maximum(-(4.0/27.0) * (qv ** 3), 0.0))
        return qv, +rv, -rv

    def _hexbin_mean(self, ax, x, y, c=None, gridsize=220, cmap='jet', vmin=0.0, vmax=2.0):
        hex_kwargs = dict(
            gridsize=gridsize,
            cmap=cmap,
            linewidths=0.1,
            vmin=vmin,
            vmax=vmax,
            mincnt=1
        )
        if c is None:
            return ax.hexbin(x, y, **hex_kwargs)
        return ax.hexbin(x, y, C=c, reduce_C_function=np.mean, **hex_kwargs)

    @staticmethod
    def _gaussian_kernel(sigma: float | None):
        if sigma is None or sigma <= 0:
            return None
        radius = int(max(1, round(3 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= np.sum(kernel)
        return kernel

    @staticmethod
    def _smooth_array(values: np.ndarray, kernel: np.ndarray | None):
        if kernel is None:
            return values
        pad = kernel.size // 2
        padded = np.pad(values, pad_width=pad, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def _pdf_curve(self, x, bins=140, clip=None, smoothing=0.0, scale=1.0, value_range=None):
        x = np.asarray(x).reshape(-1)
        if clip is not None:
            x = x[(x >= clip[0]) & (x <= clip[1])]
        if x.size == 0:
            if value_range is None:
                value_range = (0.0, 1.0)
            edges = np.linspace(value_range[0], value_range[1], bins + 1)
            centers = edges[:-1] + 0.5 * np.diff(edges)
            return centers, np.zeros_like(centers)
        if value_range is None:
            value_range = (float(np.min(x)), float(np.max(x)))
        counts, edges = np.histogram(x, bins=bins, range=value_range)
        widths = np.diff(edges)
        total = counts.sum()
        centers = edges[:-1] + widths / 2.0
        if total == 0:
            return centers, np.zeros_like(centers)
        pdf = (counts / (total * widths)) * scale
        kernel = self._gaussian_kernel(smoothing)
        if kernel is not None:
            pdf = self._smooth_array(pdf, kernel)
        return centers, pdf

    # ------------------------------------------------------------------ #
    # 1) ψ(q,r) — separate DNS and AIBM (no overlay) + legends/colorbars
    # ------------------------------------------------------------------ #
    def fig_01_psi_qr(self):
        self._prepare()
        q, r = self.q_np, self.r_np
        vmin, vmax = 0.0, float(np.percentile(np.concatenate([self.psi_true_np, self.psi_pred_np]), 99.5))

        # DNS
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hb = self._hexbin_mean(ax, r, q, self.psi_true_np, vmin=0.0, vmax=2.0)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'w--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'w--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title(r'DNS: $\psi(q,r)$'); ax.grid(True, ls=':', alpha=0.45)
        ax.set_xlim(-0.25, 0.25); ax.set_ylim(-0.5, 0.5)
        cb = fig.colorbar(hb, ax=ax, pad=0.01)
        cb.set_label(r'$\langle \psi \rangle$')
        hb.set_clim(0.0, 2.0)
        ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '01A_psi_qr_dns.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

        # AIBM
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hb = self._hexbin_mean(ax, r, q, self.psi_pred_np, vmin=0.0, vmax=2.0)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'w--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'w--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title(r'AIBM: $\psi(q,r)$'); ax.grid(True, ls=':', alpha=0.45)
        ax.set_xlim(-0.25, 0.25); ax.set_ylim(-0.5, 0.5)
        cb = fig.colorbar(hb, ax=ax, pad=0.01)
        cb.set_label(r'$\langle \psi \rangle$')
        hb.set_clim(0.0, 2.0)
        ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '01B_psi_qr_aibm.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2) Joint PDF of (q,r) — separate DNS and AIBM, with colorbars
    # ------------------------------------------------------------------ #
    def fig_02_qr_pdf(self):
        self._prepare()
        q, r = self.q_np, self.r_np

        # DNS
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hb = self._hexbin_mean(ax, r, q, c=None, gridsize=200, vmin=5.0, vmax=35.0)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'k--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'k--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title('DNS: joint PDF in $(q,r)$')
        ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.5, 0.5)
        ax.grid(True, ls=':', alpha=0.45); ax.legend(loc='upper left')
        cb = fig.colorbar(hb, ax=ax, pad=0.01)
        cb.set_label('PDF (a.u.)')
        hb.set_clim(5.0, 35.0)
        fig.savefig(self.save_dir / '02A_qr_pdf_dns.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

        # AIBM (same (q,r) cloud; still shown separately)
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hb = self._hexbin_mean(ax, r, q, c=None, gridsize=200, vmin=5.0, vmax=35.0)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'k--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'k--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title('AIBM: joint PDF in $(q,r)$')
        ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.5, 0.5)
        ax.grid(True, ls=':', alpha=0.45); ax.legend(loc='upper left')
        cb = fig.colorbar(hb, ax=ax, pad=0.01)
        cb.set_label('PDF (a.u.)')
        hb.set_clim(5.0, 35.0)
        fig.savefig(self.save_dir / '02B_qr_pdf_aibm.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3) Rotation-invariance (A: DNS, B: overlay)
    # ------------------------------------------------------------------ #
    def fig_03_rotation_invariance(self, theta_deg: float = 30.0):
        self._prepare()
        R = self._rot_z(theta_deg)

        # DNS tensors and rotated
        Q_dns = self.Q_true_np
        Q_dns_rot = self._rotate_batch(Q_dns, R)

        # Model tensors (recompute on rotated A)
        A_np = self.A.detach().cpu().numpy().astype(np.float64)
        A_rot_np = self._rotate_batch(A_np, R)
        with torch.no_grad():
            A_rot = torch.from_numpy(A_rot_np).to(self.device, dtype=self.A.dtype)
            s_r, w_r, eps_r, q_r, r_r = get_tensor_derivatives(A_rot)
            a_r = A_rot / eps_r.view(-1, 1, 1)
            s_norm_r = 0.5 * (a_r + a_r.transpose(1, 2))
            Qhat_r = self.mQ(s_norm_r, w_r)
            psi_r = self.mP(q_r, r_r)
        Q_mod = self.Q_pred_np
        Q_mod_rot = Qhat_r.detach().cpu().numpy().astype(np.float64) * \
                    psi_r.detach().cpu().numpy()[:, None, None] * \
                    (eps_r.detach().cpu().numpy()[:, None, None] ** 2)

        def ratios(Qa, Qb, tol: float = 1e-9):
            lam_a, _ = self._eig_asc(Qa)
            lam_b, _ = self._eig_asc(Qb)
            abs_a = np.abs(lam_a)
            abs_b = np.abs(lam_b)
            ratio = np.ones_like(abs_a)

            stable = abs_a > tol
            ratio[stable] = abs_b[stable] / abs_a[stable]

            near_zero_both = (~stable) & (abs_b <= tol)
            ratio[near_zero_both] = 1.0

            remaining = (~stable) & (~near_zero_both)
            if np.any(remaining):
                warnings.warn(
                    "Invariant ratio encountered |lambda| < tol for original tensor but not for rotated; "
                    "treating ratio conservatively. Inspect inputs if this persists."
                )
                ratio[remaining] = abs_b[remaining] / tol
            return ratio  # columns: [γ,β,α]

        F_dns = ratios(Q_dns, Q_dns_rot)
        F_mod = ratios(Q_mod, Q_mod_rot)

        titles = [r"$|\lambda'_\gamma|/|\lambda_\gamma|$",
                  r"$|\lambda'_\beta|/|\lambda_\beta|$",
                  r"$|\lambda'_\alpha|/|\lambda_\alpha|$"]

        # DNS only
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
        for k in range(3):
            sns.kdeplot(F_dns[:, k], ax=axs[k], bw_adjust=0.9, color="k", lw=1.8, label="DNS", clip=(0.0, 6.0))
            axs[k].set_title(titles[k]); axs[k].set_xlabel("Ratio"); axs[k].grid(True, ls=':', alpha=0.45)
            axs[k].set_yscale('linear')
            axs[k].set_xlim(0.0, 6.0)
            axs[k].set_ylim(0.0, 7.0)
        axs[0].legend()
        fig.savefig(self.save_dir / '03A_rotation_invariance_dns.png', dpi=self.DPI); plt.close(fig)

        # DNS vs AIBM overlay
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
        for k in range(3):
            sns.kdeplot(F_dns[:, k], ax=axs[k], bw_adjust=0.9, color="k", lw=1.8, label="DNS", clip=(0.0, 6.0))
            sns.kdeplot(F_mod[:, k], ax=axs[k], bw_adjust=0.9, color="C1", lw=1.8, label="AIBM", clip=(0.0, 6.0))
            axs[k].set_title(titles[k]); axs[k].set_xlabel("Ratio"); axs[k].grid(True, ls=':', alpha=0.45)
            axs[k].set_yscale('linear')
            axs[k].set_xlim(0.0, 6.0)
            axs[k].set_ylim(0.0, 35.0)
        axs[0].legend()
        fig.savefig(self.save_dir / '03B_rotation_invariance_overlay.png', dpi=self.DPI); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4) Orientation: s vs Q′ — exact panel order, ASC eigen ordering
    # ------------------------------------------------------------------ #
    def fig_04_s_vs_Q(self):
        self._prepare()
        s_np = self.s_norm.detach().cpu().numpy().astype(np.float64)
        _, Vs = self._eig_asc(s_np)
        _, Vt = self._eig_asc(self.Qhat_true_np)
        _, Vp = self._eig_asc(self.Qhat_pred_np)
        # γ,β,α indices = 0,1,2
        s_e = [Vs[:, :, 0], Vs[:, :, 1], Vs[:, :, 2]]  # [γ_s, β_s, α_s]
        t_e = [Vt[:, :, 0], Vt[:, :, 1], Vt[:, :, 2]]  # [γ_p, β_p, α_p]
        p_e = [Vp[:, :, 0], Vp[:, :, 1], Vp[:, :, 2]]  # [γ_p, β_p, α_p]

        def cosabs(U, V):
            U = self._unit_rows(U); V = self._unit_rows(V)
            return np.clip(np.abs(np.sum(U * V, axis=1)), 0.0, 1.0)

        labels = [[r'$\hat e_{\gamma_s}\!\cdot\!\hat e_{\gamma_p}$',
                   r'$\hat e_{\gamma_s}\!\cdot\!\hat e_{\beta_p}$',
                   r'$\hat e_{\gamma_s}\!\cdot\!\hat e_{\alpha_p}$'],
                  [r'$\hat e_{\beta_s}\!\cdot\!\hat e_{\gamma_p}$',
                   r'$\hat e_{\beta_s}\!\cdot\!\hat e_{\beta_p}$',
                   r'$\hat e_{\beta_s}\!\cdot\!\hat e_{\alpha_p}$'],
                  [r'$\hat e_{\alpha_s}\!\cdot\!\hat e_{\gamma_p}$',
                   r'$\hat e_{\alpha_s}\!\cdot\!\hat e_{\beta_p}$',
                   r'$\hat e_{\alpha_s}\!\cdot\!\hat e_{\alpha_p}$']]

        # DNS only
        fig, axs = plt.subplots(3, 3, figsize=(12.6, 10.6), constrained_layout=True)
        for i in range(3):
            for j in range(3):
                xs, ys = self._pdf_curve(
                    cosabs(s_e[i], t_e[j]),
                    bins=120,
                    value_range=(0.0, 1.0),
                    smoothing=1.5,
                    scale=0.5
                )
                ax = axs[i, j]
                ax.plot(xs, ys, 'k-', lw=2, label='DNS')
                ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
                ax.set_ylim(0, 2)
                ax.set_title(labels[i][j]); ax.grid(True, ls=':', alpha=0.5)
        axs[0, 0].legend(loc='upper right')
        fig.savefig(self.save_dir / '04A_sQ_align_dns.png', dpi=self.DPI); plt.close(fig)

        # DNS vs AIBM
        fig, axs = plt.subplots(3, 3, figsize=(12.6, 10.6), constrained_layout=True)
        for i in range(3):
            for j in range(3):
                ax = axs[i, j]
                xs, ys = self._pdf_curve(
                    cosabs(s_e[i], t_e[j]),
                    bins=120,
                    value_range=(0.0, 1.0),
                    smoothing=1.5,
                    scale=0.5
                )
                ax.plot(xs, ys, 'k-', lw=2, label='DNS')
                xs2, ys2 = self._pdf_curve(
                    cosabs(s_e[i], p_e[j]),
                    bins=120,
                    value_range=(0.0, 1.0),
                    smoothing=1.5,
                    scale=0.5
                )
                ax.plot(xs2, ys2, 'C1--', lw=2, label='AIBM')
                ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
                ax.set_ylim(0, 2)
                ax.set_title(labels[i][j]); ax.grid(True, ls=':', alpha=0.5)
        axs[0, 0].legend(loc='upper right')
        fig.savefig(self.save_dir / '04B_sQ_align_overlay.png', dpi=self.DPI); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5) Orientation: vorticity vs Q′ — order γ_p·ω, β_p·ω, α_p·ω
    # ------------------------------------------------------------------ #
    def fig_05_w_vs_Q(self):
        self._prepare()
        w_np = self.w.detach().cpu().numpy().astype(np.float64)
        omg = self._omega_from_w(w_np)
        _, Vt = self._eig_asc(self.Qhat_true_np)
        _, Vp = self._eig_asc(self.Qhat_pred_np)
        t_e = [Vt[:, :, 0], Vt[:, :, 1], Vt[:, :, 2]]
        p_e = [Vp[:, :, 0], Vp[:, :, 1], Vp[:, :, 2]]
        labels = [r'$\hat e_{\gamma_p}\!\cdot\!\hat\omega$',
                  r'$\hat e_{\beta_p}\!\cdot\!\hat\omega$',
                  r'$\hat e_{\alpha_p}\!\cdot\!\hat\omega$']

        def cosabs(U, V):
            U = self._unit_rows(U); V = self._unit_rows(V)
            return np.clip(np.abs(np.sum(U * V, axis=1)), 0.0, 1.0)

        # DNS only
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
        for j in range(3):
            xs, ys = self._pdf_curve(
                cosabs(omg, t_e[j]),
                bins=120,
                value_range=(0.0, 1.0),
                    smoothing=1.5,
                scale=0.5
            )
            ax = axs[j]; ax.plot(xs, ys, 'k-', lw=2, label='DNS')
            ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
            ax.set_ylim(0, 2)
            ax.set_title(labels[j]); ax.grid(True, ls=':', alpha=0.5)
        axs[0].legend(loc='upper right')
        fig.savefig(self.save_dir / '05A_wQ_align_dns.png', dpi=self.DPI); plt.close(fig)

        # DNS vs AIBM
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
        for j in range(3):
            ax = axs[j]
            xs, ys = self._pdf_curve(
                cosabs(omg, t_e[j]),
                bins=120,
                value_range=(0.0, 1.0),
                smoothing=1.2,
                scale=0.5
            )
            ax.plot(xs, ys, 'k-', lw=2, label='DNS')
            xs2, ys2 = self._pdf_curve(
                cosabs(omg, p_e[j]),
                bins=120,
                value_range=(0.0, 1.0),
                smoothing=1.2,
                scale=0.5
            )
            ax.plot(xs2, ys2, 'C1--', lw=2, label='AIBM')
            ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
            ax.set_ylim(0, 2)
            ax.set_title(labels[j]); ax.grid(True, ls=':', alpha=0.5)
        axs[0].legend(loc='upper right')
        fig.savefig(self.save_dir / '05B_wQ_align_overlay.png', dpi=self.DPI); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 6) ψ marginal PDFs — DNS-only + overlay with RMSE marker
    # ------------------------------------------------------------------ #
    def fig_06_psi_pdf(self):
        self._prepare()
        psi_t, psi_p = self.psi_true_np, self.psi_pred_np
        rmse = float(np.sqrt(np.mean((psi_p - psi_t) ** 2)))

        # DNS only
        fig, ax = plt.subplots(figsize=(6.9, 4.8))
        xs, ys = self._pdf_curve(psi_t, bins=160,
                                 clip=(np.percentile(psi_t, 0.1), np.percentile(psi_t, 99.9)),
                                 smoothing=1.5)
        ax.plot(xs, ys, 'k-', lw=2, label='DNS')
        ax.set_xlabel(r'$\psi$'); ax.set_ylabel('PDF'); ax.grid(True, ls=':', alpha=0.5)
        ax.set_title(r'DNS: marginal PDF of $\psi$'); ax.legend()
        fig.savefig(self.save_dir / '06A_psi_pdf_dns.png', dpi=self.DPI); plt.close(fig)

        # Overlay + RMSE vertical line
        fig, ax = plt.subplots(figsize=(6.9, 4.8))
        xs, ys = self._pdf_curve(psi_t, bins=160,
                                 clip=(np.percentile(psi_t, 0.1), np.percentile(psi_t, 99.9)),
                                 smoothing=1.5)
        ax.plot(xs, ys, 'k-', lw=2, label='DNS')
        xs2, ys2 = self._pdf_curve(psi_p, bins=160,
                                   clip=(np.percentile(psi_p, 0.1), np.percentile(psi_p, 99.9)),
                                   smoothing=1.5)
        ax.plot(xs2, ys2, 'C1--', lw=2, label='AIBM')
        ax.axvline(rmse, color='C3', lw=1.8, ls='-.', label=f'RMSE = {rmse:.3f}')
        ax.set_xlabel(r'$\psi$'); ax.set_ylabel('PDF'); ax.grid(True, ls=':', alpha=0.5)
        ax.set_title(r'DNS vs AIBM: marginal PDF of $\psi$'); ax.legend()
        fig.savefig(self.save_dir / '06B_psi_pdf_overlay_rmse.png', dpi=self.DPI); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 7) φ vs qε² — NO EMA; DNS-only and DNS/AIBM each with full & zoom
    # ------------------------------------------------------------------ #
    def fig_07_phi_qe2(self, nbins: int = 100, min_count: int = 1, smooth_full_sigma: float = 1.5, smooth_zoom_sigma: float = 1.5):
        """
        Fig. 07 — φ vs q ε² (Chevillard Fig. 6 recipe):
          x-axis: x = Q / σ_Q with Q = q ε² = -½ tr(A²)
          y-axis: φ(x) = E[ ||Q||_F² / σ_{||Q||_F²} | x-bin ]
        Two figure sets:
          • DNS only (full & zoom)
          • DNS vs AIBM overlay (full & zoom)
        Fixed-width bins on normalized x with optional Gaussian smoothing.
        """
        # [F12-07-01] Prepare arrays (already computed in _prepare)
        self._prepare()
        x_raw = self.X_qe2_np                                  # Q = q ε²
        y_dns_raw = self.Y_true_np                             # ||Q||² (DNS)
        y_mod_raw = self.Y_pred_np                             # ||Q||² (AIBM)

        # [F12-07-02] Normalize axes as in Chevillard Fig. 6 (std with ddof=1)
        sig_x = float(np.std(x_raw, ddof=1))
        sig_y_dns = float(np.std(y_dns_raw, ddof=1))
        sig_y_mod = float(np.std(y_mod_raw, ddof=1))

        x = x_raw / (sig_x + 1e-15)                            # Q / σ_Q
        y_dns = y_dns_raw / (sig_y_dns + 1e-15)                # ||Q||² / σ_{||Q||²} (DNS)
        y_mod = y_mod_raw / (sig_y_mod + 1e-15)                # ||Q||² / σ_{||Q||²} (AIBM)

        # [F12-07-03] Binning utility (fixed-width; returns centers, means, counts)
        def binned_mean(xv, yv, lo, hi, nbin: int):
            edges = np.linspace(lo, hi, nbin + 1)
            idx = np.clip(np.digitize(xv, edges) - 1, 0, nbin - 1)
            # Aggregate with numpy for speed
            sums = np.bincount(idx, weights=yv, minlength=nbin).astype(np.float64)
            counts = np.bincount(idx, minlength=nbin).astype(np.int64)
            with np.errstate(invalid='ignore', divide='ignore'):
                means = sums / np.maximum(counts, 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return centers, means, counts

        # [F12-07-04] Full-range domain (robust, avoid deep tails)
        x_lo_full = float(np.percentile(x, 0.5))
        x_hi_full = float(np.percentile(x, 99.5))

        # [F12-07-05] Zoom domain exactly [-1, +1] as in the paper window
        x_lo_zoom, x_hi_zoom = -1.0, 1.0

        # [F12-07-06] Compute binned conditional means
        cx_f, mu_dns_f, cnt_f = binned_mean(x, y_dns, x_lo_full, x_hi_full, nbins*10)
        _,     mu_mod_f, cntm_f = binned_mean(x, y_mod, x_lo_full, x_hi_full, nbins*10)

        cx_z, mu_dns_z, cnt_z = binned_mean(x, y_dns, x_lo_zoom, x_hi_zoom, nbins)
        _,     mu_mod_z, cntm_z = binned_mean(x, y_mod, x_lo_zoom, x_hi_zoom, nbins)

        def smooth_means(values, counts, sigma):
            kernel = self._gaussian_kernel(sigma)
            if kernel is None:
                return values
            valid_mask = counts > 0
            temp = values.copy()
            temp[~valid_mask] = 0.0
            smoothed_vals = self._smooth_array(temp, kernel)
            weights = self._smooth_array(valid_mask.astype(np.float64), kernel)
            with np.errstate(invalid='ignore', divide='ignore'):
                smoothed_vals = np.divide(smoothed_vals, np.maximum(weights, 1e-12))
            return smoothed_vals

        mu_dns_f = smooth_means(mu_dns_f, cnt_f, smooth_full_sigma)
        mu_mod_f = smooth_means(mu_mod_f, cntm_f, smooth_full_sigma)
        mu_dns_z = smooth_means(mu_dns_z, cnt_z, smooth_zoom_sigma)
        mu_mod_z = smooth_means(mu_mod_z, cntm_z, smooth_zoom_sigma)

        # [F12-07-07] Masks for minimum occupancy per bin
        mF_dns = cnt_f >= min_count
        mF_both = (cnt_f >= min_count) & (cntm_f >= min_count)

        mZ_dns = cnt_z >= min_count
        mZ_both = (cnt_z >= min_count) & (cntm_z >= min_count)

        # [F12-07-08] Plot: DNS only (FULL)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(cx_f[mF_dns], mu_dns_f[mF_dns], 'k-', lw=2.0, label='DNS')
        ax.set_xlabel(r'$Q/\sigma_Q$')
        ax.set_ylabel(r'$\langle \|Q\|_F^{2}/\sigma_{\|Q\|_F^{2}}\mid Q/\sigma_Q\rangle$')
        ax.set_title(r'DNS: $\phi$ vs. $Q/\sigma_Q$ (full range)')
        ax.grid(True, ls=':', alpha=0.55)
        ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '07A_phi_qe2_dns_full.png', dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)

        # [F12-07-09] Plot: DNS only (ZOOM, y-lims per paper style)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(cx_z[mZ_dns], mu_dns_z[mZ_dns], 'k-', lw=2.0, label='DNS')
        ax.set_xlim(x_lo_zoom, x_hi_zoom)
        ax.set_ylim(0.0, 0.25)  # per paper’s panel convention
        ax.set_xlabel(r'$Q/\sigma_Q$')
        ax.set_ylabel(r'$\langle \|Q\|_F^{2}/\sigma_{\|Q\|_F^{2}}\mid Q/\sigma_Q\rangle$')
        ax.set_title(r'DNS: $\phi$ vs. $Q/\sigma_Q$ (zoom: $[-1,1]\times[0,0.25]$)')
        ax.grid(True, ls=':', alpha=0.55)
        ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '07B_phi_qe2_dns_zoom.png', dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)

        # [F12-07-10] Plot: DNS vs AIBM overlay (FULL)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(cx_f[mF_dns],  mu_dns_f[mF_dns], 'k-',  lw=2.0, label='DNS')
        ax.plot(cx_f[mF_both], mu_mod_f[mF_both], 'C1--', lw=2.0, label='AIBM')
        ax.set_xlabel(r'$Q/\sigma_Q$')
        ax.set_ylabel(r'$\langle \|Q\|_F^{2}/\sigma_{\|Q\|_F^{2}}\mid Q/\sigma_Q\rangle$')
        ax.set_title(r'DNS vs AIBM: $\phi$ vs. $Q/\sigma_Q$ (full range)')
        ax.grid(True, ls=':', alpha=0.55)
        ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '07C_phi_qe2_overlay_full.png', dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)

        # [F12-07-11] Plot: DNS vs AIBM overlay (ZOOM)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(cx_z[mZ_dns],  mu_dns_z[mZ_dns], 'k-',  lw=2.0, label='DNS')
        ax.plot(cx_z[mZ_both], mu_mod_z[mZ_both], 'C1--', lw=2.0, label='AIBM')
        ax.set_xlim(x_lo_zoom, x_hi_zoom)
        ax.set_ylim(0.0, 0.25)
        ax.set_xlabel(r'$Q/\sigma_Q$')
        ax.set_ylabel(r'$\langle \|Q\|_F^{2}/\sigma_{\|Q\|_F^{2}}\mid Q/\sigma_Q\rangle$')
        ax.set_title(r'DNS vs AIBM: $\phi$ vs. $Q/\sigma_Q$ (zoom: $[-1,1]\times[0,0.25]$)')
        ax.grid(True, ls=':', alpha=0.55)
        ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '07D_phi_qe2_overlay_zoom.png', dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Orchestrator
    # ------------------------------------------------------------------ #
    def plot_all(self):
        self._prepare()
        self.fig_01_psi_qr()                     # 01A, 01B (legends + colorbars)
        self.fig_02_qr_pdf()                     # 02A, 02B (colorbars + legends)
        self.fig_03_rotation_invariance(30.0)    # 03A, 03B
        self.fig_04_s_vs_Q()                     # 04A, 04B (ASC order; requested layout)
        self.fig_05_w_vs_Q()                     # 05A, 05B (ASC order; requested layout)
        self.fig_06_psi_pdf()                    # 06A, 06B (RMSE line)
        self.fig_07_phi_qe2(min_count=40, smooth_full_sigma=1.5, smooth_zoom_sigma=1.5)  # 07A, 07B (full & zoom)

# ==============================================================================
# 7. MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == '__main__':
    # -------- Argument parsing --------
    parser = argparse.ArgumentParser(description="AIBM Training/Resume Script")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume (last)")
    parser.add_argument('--resume-best', type=str, default=None, help="Path to best model checkpoint to resume")
    parser.add_argument('--run-dir', type=str, default=None, help="Path to specific run directory (default: latest run)")
    parser.add_argument('--train', action='store_true', help="If set, run training (otherwise only analysis)")
    parser.add_argument('--epochs', type=int, default=400, help="Number of epochs to train for.")
    parser.add_argument('--learning-rate-q', type=float, default=1e-3, dest='learning_rate_q', help="Learning rate for the Q-direction model.")
    parser.add_argument('--learning-rate-psi', type=float, default=5e-3, dest='learning_rate_psi', help="Learning rate for the ψ-magnitude model.")
    parser.add_argument('--grad-clip', type=float, default=100.0, dest='grad_clip', help="Gradient clipping value.")
    parser.add_argument('--symm-loss-weight', type=float, default=1.0, dest='symm_loss_weight', help="Weight for symmetry penalty encouraging P̂ to remain symmetric.")
    parser.add_argument('--ema-smoothing', type=float, default=0.15, dest='ema_smoothing', help="EMA smoothing factor for history plots.")
    parser.add_argument('--log-header-interval', type=int, default=25, help="How often (in printed lines) to re-print the log header.")
    parser.add_argument('--print-every', type=int, default=1, help="Print metrics every N epochs.")
    parser.add_argument('--log-interval-type', type=str, default='epoch',
                        help="Use 'epoch' to log every --print-every epochs or 'time' to log after --log-interval-seconds seconds.")
    parser.add_argument('--log-interval-seconds', type=float, default=60.0,
                        help="When --log-interval-type is 'time', minimum wall-clock seconds between console logs.")
    parser.add_argument('--disable-profiler', action='store_true',
                        help="Disable intra-epoch timing profiler aggregation.")
    parser.add_argument('--compile-models', action='store_true',
                        help="Enable torch.compile on the neural networks for potential fused kernels.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout probability applied uniformly across the networks.")
    parser.add_argument('--dropout-type', type=str, default='dropout', choices=['dropout', 'shakeout'],
                        help="Use standard dropout or shakeout regularisation in the MLP blocks.")
    parser.add_argument('--shakeout-alpha', type=float, default=0.0,
                        help="Noise magnitude for shakeout; ignored when --dropout-type=dropout.")
    parser.add_argument('--optimizer-q', type=str, default='adamax', choices=['adamax', 'nadam'],
                        help="Optimizer for the Q-direction network (adamax or nadam).")
    parser.add_argument('--optimizer-psi', type=str, default='adamax', choices=['adamax', 'nadam'],
                        help="Optimizer for the ψ-magnitude network (adamax or nadam).")
    parser.add_argument('--weight-decay-q', type=float, default=0.01,
                        help="Weight decay coefficient for the Q-direction optimizer.")
    parser.add_argument('--weight-decay-psi', type=float, default=0.01,
                        help="Weight decay coefficient for the ψ optimizer.")
    parser.add_argument('--beta1-q', type=float, default=0.9,
                        help="β₁ for the Q-direction optimizer.")
    parser.add_argument('--beta2-q', type=float, default=0.999,
                        help="β₂ for the Q-direction optimizer.")
    parser.add_argument('--beta1-psi', type=float, default=0.9,
                        help="β₁ for the ψ optimizer.")
    parser.add_argument('--beta2-psi', type=float, default=0.999,
                        help="β₂ for the ψ optimizer.")
    parser.add_argument('--nadam-momentum-decay-q', type=float, default=0.004,
                        help="Momentum decay for NAdam when used on the Q-direction model.")
    parser.add_argument('--nadam-momentum-decay-psi', type=float, default=0.004,
                        help="Momentum decay for NAdam when used on the ψ model.")
    parser.add_argument('--chained-div-factor', type=float, default=25.0,
                        help="Divisor applied to max_lr to obtain the chained scheduler base LR.")
    parser.add_argument('--chained-final-div-factor', type=float, default=10000.0,
                        help="Divisor applied to max_lr to obtain the chained scheduler minimum LR during decay.")
    parser.add_argument('--chained-three-phase', action='store_true',
                        help="Include a cosine annealing phase between warmup and exponential decay in the chained scheduler.")
    parser.add_argument('--chained-anneal-strategy', type=str, default='cos', choices=['cos', 'linear'],
                        help="Interpolation strategy for chained warmup/anneal segments.")
    parser.add_argument('--lr-scheduler', type=str, default='linear',
                        choices=['none', 'linear', 'cosine', 'warmup_cosine', 'chained'],
                        help="Learning rate scheduler strategy.")
    parser.add_argument('--scheduler-warmup', type=float, default=0.3,
                        help="Warmup duration shared across schedulers (fraction in (0,1] or absolute epochs ≥1).")
    parser.add_argument('--bayes-opt-trials', type=int, default=0, dest='bayes_opt_trials',
                        help="Number of Bayesian optimisation trials to run before training.")
    parser.add_argument('--bayes-opt-epochs', type=int, default=50, dest='bayes_opt_epochs',
                        help="Epochs per Bayesian optimisation trial.")
    args = parser.parse_args()

    # -------- Set up RUN_DIR --------
    figs_root = pathlib.Path('figs')
    figs_root.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        RUN_DIR = pathlib.Path(args.run_dir)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
    else:
        run_dirs = sorted([d for d in figs_root.glob('run_*') if d.is_dir()])
        if args.train or not run_dirs:
            RUN_DIR = figs_root / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
            RUN_DIR.mkdir(parents=True, exist_ok=True)
        else:
            RUN_DIR = run_dirs[-1]
    print(f"Using RUN_DIR: {RUN_DIR}")

    # -------- Data Load (same as before) --------
    try:
        scipy.io.loadmat('velGrad.mat'); scipy.io.loadmat('PH.mat')
    except FileNotFoundError:
        print("Creating dummy .mat files for demonstration...")
        N = 100000
        vel_grad_dummy = np.random.randn(9, N).astype(np.float32)
        ph_dummy = np.random.randn(N, 9).astype(np.float32)
        scipy.io.savemat('velGrad.mat', {'velGrad': vel_grad_dummy}); scipy.io.savemat('PH.mat', {'PH': ph_dummy})

    full_dataset = MatlabDataset('velGrad.mat', 'PH.mat')
    train_size = int(0.8 * len(full_dataset)); val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=65536, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=65536, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

    best_bayes_trial = None
    if args.bayes_opt_trials > 0:
        try:
            best_bayes_trial = run_bayesian_optimization(
                train_loader=train_loader,
                val_loader=val_loader,
                base_args=args,
                trials=args.bayes_opt_trials,
                epochs_per_trial=args.bayes_opt_epochs
            )
            print("\nBayesian optimisation best trial:")
            for key, value in sorted(best_bayes_trial.params.items()):
                print(f"  {key}: {value}")
            print(f"  validation_euler: {best_bayes_trial.value}")

            params = best_bayes_trial.params
            args.learning_rate_q = params.get('learning_rate_Q', args.learning_rate_q)
            args.learning_rate_psi = params.get('learning_rate_psi', args.learning_rate_psi)
            args.grad_clip = params.get('grad_clip', args.grad_clip)
            args.dropout = params.get('dropout', args.dropout)
            args.lr_scheduler = params.get('lr_scheduler', args.lr_scheduler)
            args.scheduler_warmup = params.get('scheduler_warmup', args.scheduler_warmup)
        except RuntimeError as opt_err:
            print(f"Bayesian optimisation skipped: {opt_err}")
        except Exception as opt_generic:
            print(f"Bayesian optimisation failed: {opt_generic}")

    model_Q = TBNN_Q_direction(dropout_p=args.dropout,
                               dropout_type=args.dropout_type,
                               shakeout_alpha=args.shakeout_alpha)
    model_psi = FCNN_psi_magnitude(dropout_p=args.dropout,
                                   dropout_type=args.dropout_type,
                                   shakeout_alpha=args.shakeout_alpha)

    metrics_path = str(RUN_DIR / 'metrics.csv') if args.train else None
    trainer = AIBMTrainer(
        model_Q,
        model_psi,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate_Q=args.learning_rate_q,
        learning_rate_psi=args.learning_rate_psi,
        grad_clip_value=args.grad_clip,
        symm_loss_weight=args.symm_loss_weight,
        print_every=max(1, args.print_every),
        metrics_save_path=metrics_path,
        ema_smoothing=args.ema_smoothing,
        log_header_interval=args.log_header_interval,
        dropout_p=args.dropout,
        lr_scheduler_type=args.lr_scheduler,
        scheduler_warmup=args.scheduler_warmup,
        snapshot_dir=str(RUN_DIR),
        log_interval_type=args.log_interval_type,
        log_interval_seconds=args.log_interval_seconds,
        enable_profiler=not args.disable_profiler,
        compile_models=args.compile_models,
        optimizer_name_q=args.optimizer_q,
        optimizer_name_psi=args.optimizer_psi,
        weight_decay_q=args.weight_decay_q,
        weight_decay_psi=args.weight_decay_psi,
        beta1_q=args.beta1_q,
        beta2_q=args.beta2_q,
        beta1_psi=args.beta1_psi,
        beta2_psi=args.beta2_psi,
        nadam_momentum_decay_q=args.nadam_momentum_decay_q,
        nadam_momentum_decay_psi=args.nadam_momentum_decay_psi,
        chained_div_factor=args.chained_div_factor,
        chained_final_div_factor=args.chained_final_div_factor,
        chained_three_phase=args.chained_three_phase,
        chained_anneal_strategy=args.chained_anneal_strategy
    )

    # -------- Auto-restore logic --------
    def find_latest_checkpoint(run_dir, best=True):
        tag = 'best' if best else 'last'
        files = sorted(run_dir.glob(f"snapshot_epoch*_{tag}.pt"))
        if files: return str(files[-1])
        else: return None

    # Priority: --resume-best > --resume > auto-load latest best > none
    ckpt_path = None
    if args.resume_best:
        ckpt_path = args.resume_best
    elif args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = find_latest_checkpoint(RUN_DIR, best=True)
        if not ckpt_path:  # fallback: last
            ckpt_path = find_latest_checkpoint(RUN_DIR, best=False)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Restoring model from checkpoint: {ckpt_path}")
        trainer.load_snapshot(ckpt_path)
    else:
        print("No checkpoint restored; training will start from scratch.")

    # -------- Train (if requested) --------
    if args.train or not ckpt_path:
        trainer.train()
        trainer.plot_history(save_dir=str(RUN_DIR))

    # -------- Post-training analysis --------
    print("\n" + "="*25 + " POST-TRAINING ANALYSIS " + "="*25)
    try:
        visualizer = AIBMVisualizer(trainer.model_Q, trainer.model_psi, full_dataset)
        visualizer.plot_all()
    except Exception as e:
        print(f"Could not generate visualizations due to an error: {e}")
