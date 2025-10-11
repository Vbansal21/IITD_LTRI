import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import datetime, pathlib
import scipy.io
import pandas as pd
import os
import argparse
import glob
from pathlib import Path

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
torch.set_default_dtype(torch.float64)
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
    def __init__(self, vel_grad_path, pressure_hessian_path):
        print(f"Loading data from {vel_grad_path} and {pressure_hessian_path}...")
        
        def load_mat_data(path):
            mat = scipy.io.loadmat(path)
            key = next(k for k in mat if k not in ('__header__', '__version__', '__globals__'))
            return mat[key].astype(np.float64)

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

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.A[idx], self.Q[idx]
    def get_full_dataset(self):
        """Returns the full tensors for A and Q."""
        return self.A, self.Q

# ==============================================================================
# 3. MODEL DEFINITIONS (With Hard Symmetry Constraint)
# ==============================================================================

class TBNN_Q_direction(nn.Module):
    def __init__(self, hidden_layers=[50, 100, 100, 100, 50], dropout_p=0.1):
        super().__init__()
        layers = []; input_dim = 5
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_p))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, s, w):
        s_sq = torch.einsum('bik,bkj->bij', s, s); w_sq = torch.einsum('bik,bkj->bij', w, w)
        lambda_1 = torch.einsum('bii->b', s_sq); lambda_2 = torch.einsum('bii->b', w_sq)
        s_cubed = torch.einsum('bik,bkj->bij', s_sq, s); lambda_3 = torch.einsum('bii->b', s_cubed)
        w_sq_s = torch.einsum('bik,bkj->bij', w_sq, s); lambda_4 = torch.einsum('bii->b', w_sq_s)
        w_sq_s_sq = torch.einsum('bik,bkj->bij', w_sq, s_sq); lambda_5 = torch.einsum('bii->b', w_sq_s_sq)
        invariants = torch.stack([lambda_1, lambda_2, lambda_3, lambda_4, lambda_5], dim=1)
        T = self._compute_tensor_bases(s, w, s_sq, w_sq)
        g = self.network(invariants)
        Q_hat_prime_pred = torch.einsum('bn,bnij->bij', g, T)

        raw_pred = Q_hat_prime_pred

        I3 = torch.eye(3, device=Q_hat_prime_pred.device, dtype=Q_hat_prime_pred.dtype)
        trace = torch.einsum('bii->b', Q_hat_prime_pred)
        Q_hat_prime_pred = Q_hat_prime_pred - trace[:, None, None] * I3 / 3.0

        Q_hat_prime_pred = 0.5 * (Q_hat_prime_pred + Q_hat_prime_pred.transpose(-2, -1))

        norm_Q_pred_sq = torch.sum(Q_hat_prime_pred**2, dim=(1, 2), keepdim=True)
        norm_Q_pred = torch.sqrt(norm_Q_pred_sq + 1e-16)
        return Q_hat_prime_pred / norm_Q_pred, raw_pred

    def set_dropout_p(self, p: float):
        p_clamped = float(np.clip(p, 0.0, 1.0))
        for module in self.network:
            if isinstance(module, nn.Dropout):
                module.p = p_clamped

    def _compute_tensor_bases(self, s, w, s_sq, w_sq):
        I = torch.eye(3, device=s.device, dtype=s.dtype).unsqueeze(0).expand(s.shape[0], -1, -1)
        T1 = s; T2 = torch.einsum('bik,bkj->bij', s, w) - torch.einsum('bik,bkj->bij', w, s)
        T3 = s_sq - torch.einsum('bii->b', s_sq).view(-1, 1, 1) / 3 * I
        T4 = w_sq - torch.einsum('bii->b', w_sq).view(-1, 1, 1) / 3 * I
        T5 = torch.einsum('bik,bkj->bij', w, s_sq) - torch.einsum('bik,bkj->bij', s_sq, w)
        sw2 = torch.einsum('bik,bkj->bij', s, w_sq); w2s = torch.einsum('bik,bkj->bij', w_sq, s)
        T6 = w2s + sw2 - 2./3. * torch.einsum('bii->b', sw2).view(-1, 1, 1) * I
        
        ws = torch.einsum('bik,bkj->bij', w, s)
        sw = torch.einsum('bik,bkj->bij', s, w)
        T7 = torch.einsum('bik,bkj->bij', ws, w_sq) - torch.einsum('bik,bkj->bij', w_sq, sw)
        s2w = torch.einsum('bik,bkj->bij', s_sq, w)
        T8 = torch.einsum('bik,bkj->bij', sw, s_sq) - torch.einsum('bik,bkj->bij', s_sq, ws)
        w2s2 = torch.einsum('bik,bkj->bij', w_sq, s_sq)
        s2w2 = torch.einsum('bik,bkj->bij', s_sq, w_sq)
        T9 = w2s2 + s2w2 - 2./3. * torch.einsum('bii->b', s2w2).view(-1,1,1) * I
        ws2 = torch.einsum('bik,bkj->bij', w, s_sq)
        T10 = torch.einsum('bik,bkj->bij', ws2, w_sq) - torch.einsum('bik,bkj->bij', w_sq, s2w)
        
        return torch.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], dim=1)

class FCNN_psi_magnitude(nn.Module):
    def __init__(self, hidden_layers=[50, 80, 50], dropout_p=0.1):
        super().__init__()
        layers = []; input_dim = 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_p))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, q, r):
        inputs = torch.stack([q, r], dim=1)
        # return F.relu(self.network(inputs).squeeze())
        return nn.LeakyReLU(0.1)(self.network(inputs).squeeze())
        # return self.network(inputs).squeeze()

    def set_dropout_p(self, p: float):
        p_clamped = float(np.clip(p, 0.0, 1.0))
        for module in self.network:
            if isinstance(module, nn.Dropout):
                module.p = p_clamped

# ==============================================================================
# 4. LOSS FUNCTIONS (With Physics Constraints and Log-Cosh)
# ==============================================================================
def euler_angle_loss(Q_hat_prime_pred, Q_hat_prime_true, s_true):
    try:
        # e_s_vecs: (batch,3,3) real orthogonal matrix with columns being eigenvectors
        # of s_true in ascending eigenvalue order, i.e., [λ₁≤λ₂≤λ₃] from
        # torch.linalg.eigh(). Used for calculating strain-rate eigenframe alignment.
        _, e_s_vecs = torch.linalg.eigh(s_true)
        e_gamma_s, e_beta_s, e_alpha_s = e_s_vecs[:,:,0], e_s_vecs[:,:,1], e_s_vecs[:,:,2]
        
        _, e_p_vecs_true = torch.linalg.eigh(Q_hat_prime_true)
        e_gamma_p_true, e_beta_p_true, e_alpha_p_true = e_p_vecs_true[:,:,0], e_p_vecs_true[:,:,1], e_p_vecs_true[:,:,2]
        _, e_p_vecs_pred = torch.linalg.eigh(Q_hat_prime_pred)
        e_gamma_p_pred, e_beta_p_pred, e_alpha_p_pred = e_p_vecs_pred[:,:,0], e_p_vecs_pred[:,:,1], e_p_vecs_pred[:,:,2]
        
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

def log_cosh_loss(y_pred, y_true):
    err = y_true - y_pred
    return torch.mean(torch.log(torch.cosh(err) + 1e-12))

def symmetry_loss(raw_pred_tensor):
    return torch.mean((raw_pred_tensor - raw_pred_tensor.transpose(-2, -1))**2)

# ==============================================================================
# Dropout scheduling utilities
# ==============================================================================
class DropoutScheduler:
    """Utility to update dropout probabilities during training."""
    def __init__(self, config: dict | None, total_epochs: int):
        cfg = dict(config or {})
        schedule_type = cfg.get('type', 'constant').lower()
        alias_map = {'anealingcosinedecay': 'annealingcosinedecay'}
        schedule_type = alias_map.get(schedule_type, schedule_type)
        self.schedule_type = schedule_type
        self.total_epochs = max(1, int(total_epochs))
        self.initial = float(cfg.get('initial', cfg.get('value', 0.1)))
        self.final = float(cfg.get('final', self.initial))
        self.warmup_epochs = int(cfg.get('warmup_epochs', 0))
        self.min_p = float(cfg.get('min', 0.0))
        self.max_p = float(cfg.get('max', 1.0))
        if self.min_p > self.max_p:
            self.min_p, self.max_p = self.max_p, self.min_p
        valid_modes = {'constant', 'annealingcosinedecay', 'annealing_cosine', 'cosine', 'linear'}
        if self.schedule_type not in valid_modes:
            raise ValueError(f"Unsupported dropout schedule '{schedule_type}'. Valid options: {valid_modes}")

    def value(self, epoch_idx: int) -> float:
        epoch = max(0, int(epoch_idx))
        base_value: float
        if self.schedule_type == 'constant':
            base_value = self.initial
        elif self.schedule_type in {'annealingcosinedecay', 'annealing_cosine', 'cosine'}:
            adjusted_total = max(self.total_epochs - self.warmup_epochs, 1)
            if epoch < self.warmup_epochs:
                base_value = self.initial
            else:
                progress = min(epoch - self.warmup_epochs, adjusted_total)
                cosine_term = 0.5 * (1 + math.cos(math.pi * progress / adjusted_total))
                base_value = self.final + (self.initial - self.final) * cosine_term
        else:  # linear schedule
            total = max(self.total_epochs - 1, 1)
            progress = min(epoch, total) / total
            base_value = self.initial + (self.final - self.initial) * progress
        return float(np.clip(base_value, self.min_p, self.max_p))

# ==============================================================================
# 5. TRAINING PIPELINE (FP64, Grad Clipping)
# ==============================================================================
class AIBMTrainer:
    def __init__(self, model_Q, model_psi, train_loader, val_loader,
                epochs=400, learning_rate_Q=1e-3, learning_rate_psi=5e-3,
                grad_clip_value=5.0, symm_loss_weight=0.5,
                print_every=1, metrics_save_path=None,
                ema_smoothing=0.15, log_header_interval=100,
                optimizer_betas=(0.95, 0.9995), optimizer_weight_decay=1e-2,
                optimizer_eps=1e-8, dropout_schedule_config=None,
                snapshot_dir=None):
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_Q = model_Q.to(self.device, dtype=torch.float64)
        self.model_psi = model_psi.to(self.device, dtype=torch.float64)
        self.train_loader = train_loader
        self.val_loader = val_loader
        if len(optimizer_betas) != 2:
            raise ValueError("optimizer_betas must be a tuple of length 2.")
        self.optimizer_Q = torch.optim.AdamW(
            self.model_Q.parameters(),
            lr=learning_rate_Q,
            betas=optimizer_betas,
            weight_decay=optimizer_weight_decay,
            eps=optimizer_eps
        )
        self.optimizer_psi = torch.optim.AdamW(
            self.model_psi.parameters(),
            lr=learning_rate_psi,
            betas=optimizer_betas,
            weight_decay=optimizer_weight_decay,
            eps=optimizer_eps
        )
        self.scheduler_Q = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_Q, T_max=epochs, eta_min=learning_rate_Q/25)
        self.scheduler_psi = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_psi, T_max=epochs, eta_min=learning_rate_psi/25)
        self.grad_clip_value = grad_clip_value
        self.symm_loss_weight = symm_loss_weight
        self.print_every = print_every
        self.metrics_save_path = metrics_save_path
        self.history = []
        self.train_metrics = []
        self.val_metrics = []
        self.best = {}
        self.snapshot_dir = str(snapshot_dir) if snapshot_dir else None
        self.dropout_scheduler = DropoutScheduler(dropout_schedule_config, self.epochs)
        self.current_dropout_p = self.dropout_scheduler.value(0)
        self._apply_dropout(self.current_dropout_p)

        # For EMA smoothing and logging
        self.ema_alpha = ema_smoothing
        self.log_header_interval = max(1, log_header_interval)
        self._log_line_count = 0

    def train(self, epochs=None):
        epochs = epochs if epochs else self.epochs
        header = (
            f"{'Epoch':>5} | {'train_euler':>13} | {'val_euler':>13} | "
            f"{'train_psi_rmse':>16} | {'val_psi_rmse':>16} | "
            f"{'ratio':>8} | {'t_train(s)':>11} | {'t_val(s)':>11} | {'t_total(s)':>11}"
        )
        print('\n' + header)
        print('-' * len(header))
        import time

        for epoch in range(epochs):
            epoch_idx = epoch + 1

            current_dropout = self.dropout_scheduler.value(epoch_idx - 1)
            self._apply_dropout(current_dropout)

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
                'time': train_time,
                'dropout_p': current_dropout
            }

            # ---------- Validation ----------
            self.model_Q.eval()
            self.model_psi.eval()
            with torch.no_grad():
                val_start = time.time()
                val_stats = self._epoch_pass(self.val_loader, mode='val')
                val_time = time.time() - val_start
            val_stats = {**val_stats, 'time': val_time, 'dropout_p': current_dropout}

            # ---------- Scheduler (per-epoch) ----------
            self.scheduler_Q.step()
            self.scheduler_psi.step()

            # ---------- Record ----------
            flat_record = {'epoch': epoch_idx}
            flat_record.update({f"train_{k}": v for k, v in train_stats.items()})
            flat_record.update({f"val_{k}": v for k, v in val_stats.items()})
            self.history.append(flat_record)
            self.train_metrics.append({'epoch': epoch_idx, **train_stats})
            self.val_metrics.append({'epoch': epoch_idx, **val_stats})

            ratio = val_stats['euler'] / train_stats['euler'] if train_stats['euler'] != 0 else float('nan')
            train_time = train_stats['time']
            val_time = val_stats['time']
            total_time = train_time + val_time

            if (epoch_idx % self.print_every == 0) or (epoch == 0):
                if self._log_line_count % self.log_header_interval == 0 and self._log_line_count != 0:
                    print('\n' + header)
                    print('-' * len(header))
                print(
                    f"{epoch_idx:5d} | "
                    f"{self._format_sci(train_stats['euler']):>13} | "
                    f"{self._format_sci(val_stats['euler']):>13} | "
                    f"{self._format_sci(train_stats['psi_rmse']):>16} | "
                    f"{self._format_sci(val_stats['psi_rmse']):>16} | "
                    f"{self._format_sci(ratio):>8} | "
                    f"{self._format_sci(train_time):>11} | "
                    f"{self._format_sci(val_time):>11} | "
                    f"{self._format_sci(total_time):>11}"
                )
                self._log_line_count += 1

            # ---------- Best tracking / snapshots ----------
            best_val_euler = self.best.get('euler', float('inf'))
            if val_stats['euler'] < best_val_euler:
                self.best = {'epoch': epoch_idx, **val_stats}
                if self.snapshot_dir:
                    self.save_snapshot(epoch_idx, val_stats, save_dir=self.snapshot_dir, tag='best')
            # Save rolling last snapshot
            if self.snapshot_dir:
                self.save_snapshot(epoch_idx, val_stats, save_dir=self.snapshot_dir, tag='last')

        if self.best:
            print("\nBest validation metrics (by Euler loss):")
            print(self.best)

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

    def _apply_dropout(self, p: float):
        p = float(np.clip(p, 0.0, 1.0))
        if hasattr(self.model_Q, 'set_dropout_p'):
            self.model_Q.set_dropout_p(p)
        else:
            for module in self.model_Q.modules():
                if isinstance(module, nn.Dropout):
                    module.p = p
        if hasattr(self.model_psi, 'set_dropout_p'):
            self.model_psi.set_dropout_p(p)
        else:
            for module in self.model_psi.modules():
                if isinstance(module, nn.Dropout):
                    module.p = p
        self.current_dropout_p = p

    def _epoch_pass(self, loader, mode='train'):
        losses_euler, losses_L1, losses_L2, losses_L3, losses_symm = [], [], [], [], []
        rmse_Q, rmse_psi = [], []
        psi_r2s = []
        losses_psi, losses_psi_mse = [], []
        N = 0

        for A_batch, Q_batch in loader:
            A_batch = A_batch.to(self.device, dtype=torch.float64)
            Q_batch = Q_batch.to(self.device, dtype=torch.float64)

            s, w, eps, q, r = get_tensor_derivatives(A_batch)
            _, Q_hat_t, psi_t = process_ground_truth_Q(Q_batch, eps)
            a = A_batch / eps.unsqueeze(-1).unsqueeze(-1)
            s_norm = 0.5 * (a + a.transpose(1, 2))

            Q_hat_p, raw_Q_p = self.model_Q(s_norm, w)
            psi_p = self.model_psi(q, r)

            # Euler angle and component losses
            loss_euler, euler_dict = euler_angle_loss(Q_hat_p, Q_hat_t, s_norm)
            loss_euler_raw, euler_dict_raw = euler_angle_loss(raw_Q_p, Q_hat_t, s_norm)
            sym_loss = symmetry_loss(raw_Q_p)

            losses_euler.append((loss_euler.detach().item() + loss_euler_raw.detach().item()) / 2)
            losses_L1.append((euler_dict.get('L1', 0) + euler_dict_raw.get('L1', 0)) / 2)
            losses_L2.append((euler_dict.get('L2', 0) + euler_dict_raw.get('L2', 0)) / 2)
            losses_L3.append((euler_dict.get('L3', 0) + euler_dict_raw.get('L3', 0)) / 2)
            losses_symm.append(sym_loss.detach().item())

            # RMSE and R2
            q_rmse = torch.sqrt(F.mse_loss(Q_hat_p, Q_hat_t))
            mse_psi = F.mse_loss(psi_p, psi_t)
            psi_rmse = torch.sqrt(mse_psi)
            logcosh_loss_val = log_cosh_loss(psi_p, psi_t)

            rmse_Q.append(q_rmse.detach().item())
            rmse_psi.append(psi_rmse.detach().item())
            losses_psi.append(logcosh_loss_val.detach().item())
            losses_psi_mse.append(mse_psi.detach().item())
            psi_r2s.append(self._r2(psi_p, psi_t))

            N += 1

            # Backprop only if train
            if mode == 'train':
                self.optimizer_Q.zero_grad(set_to_none=True)
                self.optimizer_psi.zero_grad(set_to_none=True)

                total_q_loss = (loss_euler*np.exp(-N) + loss_euler_raw*np.exp(N))/(2*(np.exp(-N)+np.exp(N))) + self.symm_loss_weight * sym_loss
                total_q_loss.backward()
                logcosh_loss_val.backward()

                torch.nn.utils.clip_grad_norm_(self.model_Q.parameters(), self.grad_clip_value)
                torch.nn.utils.clip_grad_norm_(self.model_psi.parameters(), self.grad_clip_value)
                self.optimizer_Q.step()
                self.optimizer_psi.step()

        # Aggregated metrics
        res = {
            'euler': np.mean(losses_euler),
            'L1': np.mean(losses_L1),
            'L2': np.mean(losses_L2),
            'L3': np.mean(losses_L3),
            'symm': np.mean(losses_symm),
            'Q_rmse': np.mean(rmse_Q),
            'psi_rmse': np.mean(rmse_psi),
            'psi_logcosh': np.mean(losses_psi),
            'psi_mse': np.mean(losses_psi_mse),
            'psi_r2': np.mean(psi_r2s),
        }
        return res

    def _r2(self, pred, true):
        pred, true = pred.flatten().detach().cpu().numpy(), true.flatten().detach().cpu().numpy()
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - np.mean(true))**2) + 1e-8
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

        # Panel 3: Q metrics
        axs[1, 0].plot(epochs, df['Q_rmse'], 'C5-', lw=2, label='Q RMSE')
        axs[1, 0].plot(epochs, ema(df['Q_rmse'], self.ema_alpha), 'C5--', lw=1.5, label='Q RMSE EMA')
        if 'symm' in df.columns:
            axs[1, 0].plot(epochs, df['symm'], 'C9-', lw=1.2, alpha=0.8, label='Symmetry loss')
        if 'hess_rmse' in df.columns:
            axs[1, 0].plot(epochs, df['hess_rmse'], 'C10-', lw=1.2, alpha=0.8, label='Max Hessian eig.')
        axs[1, 0].set_ylabel('Q metrics')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].grid(True, ls=':', alpha=0.6)
        axs[1, 0].legend()

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
        """Save model/optimizer states with optional tag (e.g., 'best', 'last')"""
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
        if tag: fname += f"_{tag}"
        fname += ".pt"
        torch.save(state, os.path.join(save_dir, fname))

    def load_snapshot(self, snapshot_path, strict=True):
        """Restore a previously saved snapshot"""
        state = torch.load(snapshot_path, map_location=self.device, weights_only=False)
        self.model_Q.load_state_dict(state['model_Q'], strict=strict)
        self.model_psi.load_state_dict(state['model_psi'], strict=strict)
        self.optimizer_Q.load_state_dict(state['optimizer_Q'])
        self.optimizer_psi.load_state_dict(state['optimizer_psi'])
        return state


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
            Qhat_pred, _ = self.mQ(s_norm, w)
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
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.5, 0.5)
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
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.5, 0.5)
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
        hb = self._hexbin_mean(ax, r, q, c=None, gridsize=200, vmin=0.0, vmax=2.0)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'k--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'k--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title('DNS: joint PDF in $(q,r)$')
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.5, 0.5)
        ax.grid(True, ls=':', alpha=0.45); ax.legend(loc='upper left')
        cb = fig.colorbar(hb, ax=ax, pad=0.01)
        cb.set_label('PDF (a.u.)')
        hb.set_clim(5.0, 40.0)
        fig.savefig(self.save_dir / '02A_qr_pdf_dns.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

        # AIBM (same (q,r) cloud; still shown separately)
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hb = self._hexbin_mean(ax, r, q, c=None, gridsize=200, vmin=0.0, vmax=2.0)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'k--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'k--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title('AIBM: joint PDF in $(q,r)$')
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.5, 0.5)
        ax.grid(True, ls=':', alpha=0.45); ax.legend(loc='upper left')
        cb = fig.colorbar(hb, ax=ax, pad=0.01)
        cb.set_label('PDF (a.u.)')
        hb.set_clim(5.0, 40.0)
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
            Qhat_r, _ = self.mQ(s_norm_r, w_r)
            psi_r = self.mP(q_r, r_r)
        Q_mod = self.Q_pred_np
        Q_mod_rot = Qhat_r.detach().cpu().numpy().astype(np.float64) * \
                    psi_r.detach().cpu().numpy()[:, None, None] * \
                    (eps_r.detach().cpu().numpy()[:, None, None] ** 2)

        def ratios(Qa, Qb):
            lam_a, _ = self._eig_asc(Qa)
            lam_b, _ = self._eig_asc(Qb)
            return np.abs(lam_b / np.maximum(np.abs(lam_a), 1e-12))  # columns: [γ,β,α]

        F_dns = ratios(Q_dns, Q_dns_rot)
        F_mod = ratios(Q_mod, Q_mod_rot)

        titles = [r"$|\lambda'_\gamma|/|\lambda_\gamma|$",
                  r"$|\lambda'_\beta|/|\lambda_\beta|$",
                  r"$|\lambda'_\alpha|/|\lambda_\alpha|$"]

        # DNS only
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
        for k in range(3):
            sns.kdeplot(F_dns[:, k], ax=axs[k], bw_adjust=0.9, color="k", lw=1.8, label="DNS")
            axs[k].set_title(titles[k]); axs[k].set_xlabel("Ratio"); axs[k].grid(True, ls=':', alpha=0.45)
            axs[k].set_yscale('linear')
            axs[k].set_ylim(0.0, 1.0e6)
        axs[0].legend()
        fig.savefig(self.save_dir / '03A_rotation_invariance_dns.png', dpi=self.DPI); plt.close(fig)

        # DNS vs AIBM overlay
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
        for k in range(3):
            sns.kdeplot(F_dns[:, k], ax=axs[k], bw_adjust=0.9, color="k", lw=1.8, label="DNS")
            sns.kdeplot(F_mod[:, k], ax=axs[k], bw_adjust=0.9, color="C1", lw=1.8, label="AIBM")
            axs[k].set_title(titles[k]); axs[k].set_xlabel("Ratio"); axs[k].grid(True, ls=':', alpha=0.45)
            axs[k].set_yscale('linear')
            axs[k].set_ylim(0.0, 1.0e6)
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
    parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train for.")
    parser.add_argument('--learning-rate-q', type=float, default=5e-3, dest='learning_rate_q', help="Learning rate for the Q-direction model.")
    parser.add_argument('--learning-rate-psi', type=float, default=25e-3, dest='learning_rate_psi', help="Learning rate for the ψ-magnitude model.")
    parser.add_argument('--grad-clip', type=float, default=5.0, dest='grad_clip', help="Gradient clipping value.")
    parser.add_argument('--symm-loss-weight', type=float, default=0.5, dest='symm_loss_weight', help="Weight for symmetry loss term.")
    parser.add_argument('--ema-smoothing', type=float, default=0.15, dest='ema_smoothing', help="EMA smoothing factor for history plots.")
    parser.add_argument('--log-header-interval', type=int, default=30, help="How often (in printed lines) to re-print the log header.")
    parser.add_argument('--print-every', type=int, default=10, help="Print metrics every N epochs.")
    parser.add_argument('--optimizer-beta1', type=float, default=0.95, dest='optimizer_beta1', help="AdamW β₁.")
    parser.add_argument('--optimizer-beta2', type=float, default=0.9995, dest='optimizer_beta2', help="AdamW β₂.")
    parser.add_argument('--optimizer-weight-decay', type=float, default=1e-2, dest='optimizer_weight_decay', help="AdamW weight decay.")
    parser.add_argument('--optimizer-eps', type=float, default=1e-8, dest='optimizer_eps', help="AdamW epsilon.")
    parser.add_argument('--dropout-schedule', type=str, default='constant',
                        choices=['constant', 'annealing_cosine', 'linear'],
                        help="Dropout scheduling strategy.")
    parser.add_argument('--dropout-initial', type=float, default=0.1, dest='dropout_initial', help="Initial dropout probability.")
    parser.add_argument('--dropout-final', type=float, default=0.1, dest='dropout_final', help="Final dropout probability.")
    parser.add_argument('--dropout-warmup', type=int, default=0, dest='dropout_warmup', help="Warmup epochs before applying decay schedules.")
    parser.add_argument('--dropout-min', type=float, default=0.0, dest='dropout_min', help="Minimum allowable dropout probability.")
    parser.add_argument('--dropout-max', type=float, default=1.0, dest='dropout_max', help="Maximum allowable dropout probability.")
    parser.add_argument('--bayes-opt-trials', type=int, default=0, dest='bayes_opt_trials', help="Number of Bayesian optimisation trials to run before training.")
    parser.add_argument('--bayes-opt-epochs', type=int, default=50, dest='bayes_opt_epochs', help="Epochs per Bayesian optimisation trial.")
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
        vel_grad_dummy = np.random.randn(9, N).astype(np.float64)
        ph_dummy = np.random.randn(N, 9).astype(np.float64)
        scipy.io.savemat('velGrad.mat', {'velGrad': vel_grad_dummy}); scipy.io.savemat('PH.mat', {'PH': ph_dummy})

    full_dataset = MatlabDataset('velGrad.mat', 'PH.mat')
    train_size = int(0.8 * len(full_dataset)); val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16384, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    best_bayes_trial = None
    if args.bayes_opt_trials > 0:
        try:
            best_bayes_trial = run_bayesian_optimization(
                train_loader,
                val_loader,
                base_args=args,
                trials=args.bayes_opt_trials,
                epochs_per_trial=args.bayes_opt_epochs
            )
            best_params = best_bayes_trial.params
            print("\nBayesian optimisation best trial:")
            for key, value in sorted(best_params.items()):
                print(f"  {key}: {value}")
            print(f"  validation_euler: {best_bayes_trial.value}")

            args.learning_rate_q = best_params.get('learning_rate_Q', args.learning_rate_q)
            args.learning_rate_psi = best_params.get('learning_rate_psi', args.learning_rate_psi)
            args.grad_clip = best_params.get('grad_clip', args.grad_clip)
            args.symm_loss_weight = best_params.get('symm_weight', args.symm_loss_weight)
            args.optimizer_beta1 = best_params.get('beta1', args.optimizer_beta1)
            args.optimizer_beta2 = best_params.get('beta2', args.optimizer_beta2)
            args.optimizer_weight_decay = best_params.get('weight_decay', args.optimizer_weight_decay)
            args.dropout_schedule = best_params.get('dropout_schedule', args.dropout_schedule)
            args.dropout_initial = best_params.get('dropout_initial', args.dropout_initial)
            args.dropout_final = best_params.get('dropout_final', args.dropout_final)
            args.dropout_warmup = best_params.get('dropout_warmup', args.dropout_warmup)
        except RuntimeError as opt_err:
            print(f"Bayesian optimisation skipped: {opt_err}")
        except Exception as opt_generic:
            print(f"Bayesian optimisation failed due to: {opt_generic}")

    dropout_config = {
        'type': args.dropout_schedule,
        'initial': args.dropout_initial,
        'final': args.dropout_final,
        'warmup_epochs': args.dropout_warmup,
        'min': args.dropout_min,
        'max': args.dropout_max
    }

    model_Q = TBNN_Q_direction(dropout_p=args.dropout_initial)
    model_psi = FCNN_psi_magnitude(dropout_p=args.dropout_initial)

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
        optimizer_betas=(args.optimizer_beta1, args.optimizer_beta2),
        optimizer_weight_decay=args.optimizer_weight_decay,
        optimizer_eps=args.optimizer_eps,
        dropout_schedule_config=dropout_config,
        snapshot_dir=str(RUN_DIR)
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


def run_bayesian_optimization(train_loader, val_loader, base_args, trials=10, epochs_per_trial=20):
    """
    Launch Bayesian optimisation over a broad hyperparameter space.
    Returns the best Optuna trial (or raises if Optuna is unavailable).
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Bayesian optimisation requested, but Optuna is not installed. "
            "Install it via `pip install optuna` and retry."
        ) from exc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(trial: "optuna.trial.Trial") -> float:
        seed_offset = trial.number
        torch.manual_seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)

        lr_q = trial.suggest_float('learning_rate_Q', 1e-4, 1e-2, log=True)
        lr_psi = trial.suggest_float('learning_rate_psi', 1e-4, 5e-2, log=True)
        grad_clip = trial.suggest_float('grad_clip', 1.0, 20.0)
        symm_weight = trial.suggest_float('symm_weight', 0.1, 2.0)
        beta1 = trial.suggest_float('beta1', 0.85, 0.99)
        beta2 = trial.suggest_float('beta2', 0.95, 0.9999)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        dropout_initial = trial.suggest_float('dropout_initial', 0.05, 0.35)
        dropout_final = trial.suggest_float('dropout_final', 0.01, 0.35)
        dropout_schedule = trial.suggest_categorical('dropout_schedule', ['constant', 'annealing_cosine', 'linear'])
        dropout_warmup = trial.suggest_int('dropout_warmup', 0, max(0, epochs_per_trial // 2))

        dropout_cfg = {
            'type': dropout_schedule,
            'initial': dropout_initial,
            'final': dropout_final,
            'warmup_epochs': dropout_warmup
        }

        model_Q = TBNN_Q_direction(dropout_p=dropout_initial).to(device, dtype=torch.float64)
        model_psi = FCNN_psi_magnitude(dropout_p=dropout_initial).to(device, dtype=torch.float64)

        trainer = AIBMTrainer(
            model_Q, model_psi,
            train_loader, val_loader,
            epochs=epochs_per_trial,
            learning_rate_Q=lr_q,
            learning_rate_psi=lr_psi,
            grad_clip_value=grad_clip,
            symm_loss_weight=symm_weight,
            print_every=max(1, epochs_per_trial // 5),
            metrics_save_path=None,
            ema_smoothing=base_args.ema_smoothing,
            log_header_interval=max(1, base_args.log_header_interval),
            optimizer_betas=(beta1, beta2),
            optimizer_weight_decay=weight_decay,
            optimizer_eps=base_args.optimizer_eps,
            dropout_schedule_config=dropout_cfg,
            snapshot_dir=None
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train(epochs=epochs_per_trial)

        best_metric = trainer.best.get('euler', float('inf'))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_metric

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_trial
