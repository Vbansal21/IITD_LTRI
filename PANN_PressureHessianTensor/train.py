import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
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

plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.autolayout": True,
    "font.size": 13
})

# --- Configuration ---
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
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
    out = np.zeros_like(arr, dtype=np.float32)
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
    rv = (2.0 / 3.0) * np.power(-qv, 1.5)  # ← no √3 factor
    return qv, +rv, -rv

def hexbin_with_mean(ax, x, y, c, gridsize=200, vmin=None, vmax=None):
    return ax.hexbin(
        x, y, C=c,
        gridsize=gridsize,
        reduce_C_function=np.mean,
        cmap="viridis",
        linewidths=0.1,
        vmin=vmin, vmax=vmax
    )

# ==============================================================================
# 2. DATASET CLASS FOR .MAT FILES
# ==============================================================================

class MatlabDataset(torch.utils.data.Dataset):
    def __init__(self, vel_grad_path, pressure_hessian_path):
        print(f"Loading data from {vel_grad_path} and {pressure_hessian_path}...")
        
        def load_mat_data(path):
            mat = scipy.io.loadmat(path)
            key = next(k for k in mat if k not in ('__header__', '__version__', '__globals__'))
            return mat[key].astype(np.float32)

        vel_grad_data = load_mat_data(vel_grad_path)
        if vel_grad_data.shape[0] == 9: vel_grad_data = vel_grad_data.T
        self.A = torch.from_numpy(vel_grad_data).view(-1, 3, 3)

        ph_data = load_mat_data(pressure_hessian_path)
        if ph_data.shape[0] == 9: ph_data = ph_data.T
        raw_P = torch.from_numpy(ph_data).view(-1, 3, 3)

        trace_P = torch.einsum('bii->b', raw_P).unsqueeze(-1).unsqueeze(-1)
        self.Q = raw_P - (trace_P / 3.0) * torch.eye(3).unsqueeze(0)

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
            layers.append(nn.Linear(input_dim, hidden_dim)); layers.append(nn.LeakyReLU(0.1)); layers.append(nn.Dropout(dropout_p)); input_dim = hidden_dim
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
        
        I3 = torch.eye(3, device=Q_hat_prime_pred.device)
        trace = torch.einsum('bii->b', Q_hat_prime_pred)
        Q_hat_prime_pred = Q_hat_prime_pred - trace[:, None, None] * I3 / 3.0
        
        Q_hat_prime_pred = 0.5 * (Q_hat_prime_pred + Q_hat_prime_pred.transpose(-2, -1))
        
        norm_Q_pred_sq = torch.sum(Q_hat_prime_pred**2, dim=(1, 2), keepdim=True)
        norm_Q_pred = torch.sqrt(norm_Q_pred_sq + 1e-16)
        return Q_hat_prime_pred / norm_Q_pred, raw_pred

    def _compute_tensor_bases(self, s, w, s_sq, w_sq):
        I = torch.eye(3, device=s.device).unsqueeze(0).expand(s.shape[0], -1, -1)
        T1 = s; T2 = torch.einsum('bik,bkj->bij', s, w) - torch.einsum('bik,bkj->bij', w, s)
        T3 = s_sq - torch.einsum('bii->b', s_sq).view(-1, 1, 1) / 3 * I
        T4 = w_sq - torch.einsum('bii->b', w_sq).view(-1, 1, 1) / 3 * I
        T5 = torch.einsum('bik,bkj->bij', w, s_sq) - torch.einsum('bik,bkj->bij', s_sq, w)
        sw2 = torch.einsum('bik,bkj->bij', s, w_sq); w2s = torch.einsum('bik,bkj->bij', w_sq, s)
        T6 = w2s + sw2 - 2./3. * torch.einsum('bii->b', sw2).view(-1, 1, 1) * I
        
        ws = torch.einsum('bik,bkj->bij', w, s); sw = torch.einsum('bik,bkj->bij', s, w)
        T7 = torch.einsum('bik,bkj->bij', ws, w_sq) - torch.einsum('bik,bkj->bij', w_sq, sw)
        s2w = torch.einsum('bik,bkj->bij', s_sq, w)
        T8 = torch.einsum('bik,bkj->bij', sw, s_sq) - torch.einsum('bik,bkj->bij', s_sq, ws)
        w2s2 = torch.einsum('bik,bkj->bij', w_sq, s_sq); s2w2 = torch.einsum('bik,bkj->bij', s_sq, w_sq)
        T9 = w2s2 + s2w2 - 2./3. * torch.einsum('bii->b', s2w2).view(-1,1,1) * I
        ws2 = torch.einsum('bik,bkj->bij', w, s_sq)
        T10 = torch.einsum('bik,bkj->bij', ws2, w_sq) - torch.einsum('bik,bkj->bij', w_sq, s2w)
        
        return torch.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], dim=1)

class FCNN_psi_magnitude(nn.Module):
    def __init__(self, hidden_layers=[50, 80, 50], dropout_p=0.1):
        super().__init__()
        layers = []; input_dim = 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim)); layers.append(nn.LeakyReLU(0.1)); layers.append(nn.Dropout(dropout_p)); input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, q, r):
        inputs = torch.stack([q, r], dim=1)
        return F.relu(self.network(inputs).squeeze())

# ==============================================================================
# 4. LOSS FUNCTIONS (With Physics Constraints and Log-Cosh)
# ==============================================================================
def euler_angle_loss(Q_hat_prime_pred, Q_hat_prime_true, s_true):
    try:
        _, e_s_vecs = torch.linalg.eigh(s_true); e_gamma_s, e_beta_s, e_alpha_s = e_s_vecs[:,:,0], e_s_vecs[:,:,1], e_s_vecs[:,:,2]
        _, e_p_vecs_true = torch.linalg.eigh(Q_hat_prime_true); e_gamma_p_true, e_beta_p_true, e_alpha_p_true = e_p_vecs_true[:,:,0], e_p_vecs_true[:,:,1], e_p_vecs_true[:,:,2]
        _, e_p_vecs_pred = torch.linalg.eigh(Q_hat_prime_pred); e_gamma_p_pred, e_beta_p_pred, e_alpha_p_pred = e_p_vecs_pred[:,:,0], e_p_vecs_pred[:,:,1], e_p_vecs_pred[:,:,2]
        
        cos_zeta_true = torch.einsum('bi,bi->b', e_gamma_p_true, e_gamma_s)
        e_proj_prime_true = e_alpha_p_true - torch.einsum('bi,bi->b', e_alpha_p_true, e_gamma_s).unsqueeze(1) * e_gamma_s
        e_proj_true = e_proj_prime_true / (torch.norm(e_proj_prime_true, dim=1, keepdim=True) + 1e-8)
        cos_theta_true = torch.einsum('bi,bi->b', e_alpha_s, e_proj_true)
        e_norm_true = torch.cross(e_proj_true, e_gamma_s); cos_eta_true = torch.einsum('bi,bi->b', e_beta_p_true, e_norm_true)
        
        cos_zeta_pred = torch.einsum('bi,bi->b', e_gamma_p_pred, e_gamma_s)
        e_proj_prime_pred = e_alpha_p_pred - torch.einsum('bi,bi->b', e_alpha_p_pred, e_gamma_s).unsqueeze(1) * e_gamma_s
        e_proj_pred = e_proj_prime_pred / (torch.norm(e_proj_prime_pred, dim=1, keepdim=True) + 1e-8)
        cos_theta_pred = torch.einsum('bi,bi->b', e_alpha_s, e_proj_pred)
        e_norm_pred = torch.cross(e_proj_pred, e_gamma_s); cos_eta_pred = torch.einsum('bi,bi->b', e_beta_p_pred, e_norm_pred)
        
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
# 5. TRAINING PIPELINE (FP32, Manual Hessian, Grad Clipping)
# ==============================================================================
class AIBMTrainer:
    def __init__(self, model_Q, model_psi, train_loader, val_loader,
                epochs=400, learning_rate_Q=1e-3, learning_rate_psi=5e-3,
                hessian_freq=5, grad_clip_value=5.0, symm_loss_weight=0.5, 
                print_every=1, metrics_save_path=None):
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_Q = model_Q.to(self.device)
        self.model_psi = model_psi.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_Q = torch.optim.Adamax(self.model_Q.parameters(), lr=learning_rate_Q)
        self.optimizer_psi = torch.optim.Adamax(self.model_psi.parameters(), lr=learning_rate_psi)
        self.scheduler_Q = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_Q, T_max=epochs, eta_min=0)
        self.scheduler_psi = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_psi, T_max=epochs, eta_min=0)
        self.hessian_freq = hessian_freq
        self.grad_clip_value = grad_clip_value
        self.symm_loss_weight = symm_loss_weight
        self.print_every = print_every
        self.metrics_save_path = metrics_save_path
        self.history = []
        self.train_metrics = []
        self.val_metrics = []
        self.best = {}

        # For EMA smoothing
        self.ema_alpha = 0.15

    def train(self, epochs=None):
        epochs = epochs if epochs else self.epochs
        print('\n{:>5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}'.format(
            'Epoch', 'Euler(Q)', 'L1', 'L2', 'L3', 'Psi_RMSE', 'Hess_RMSE', 'Val/Train', 'Time(s)'))
        print('-' * 128)
        import time

        for epoch in range(epochs):
            t0 = time.time()
            # ---------- Training ----------
            self.model_Q.train(); self.model_psi.train()
            train_stats = self._epoch_pass(self.train_loader, mode='train')
            # ---------- Validation ----------
            self.model_Q.eval(); self.model_psi.eval()
            with torch.no_grad():
                val_stats = self._epoch_pass(self.val_loader, mode='val')
            # ---------- Combine ----------
            metrics = {**{'epoch': epoch+1}, **{f'train_{k}':v for k,v in train_stats.items()},
                    **{f'val_{k}':v for k,v in val_stats.items()}}
            self.history.append(metrics)
            self.train_metrics.append(train_stats)
            self.val_metrics.append(val_stats)

            # --------- Best/EMA ----------
            for key in val_stats:
                if key not in self.best or (key.endswith('rmse') and val_stats[key]<self.best[key]):
                    self.best[key] = val_stats[key]
            
            # --- Best Model Snapshots [SNAP001] ---
            # Here, "val_euler" is used as best metric (lower is better). Change as needed.
            if (len(self.val_metrics) == 1) or (val_stats['euler'] < min([v['euler'] for v in self.val_metrics[:-1]])):
                self.save_snapshot(epoch+1, val_stats, save_dir=str(RUN_DIR), tag='best')
            # Optionally, save every epoch's last snapshot (for resume training):
            self.save_snapshot(epoch+1, val_stats, save_dir=str(RUN_DIR), tag='last')

            # ---------- Print ----------
            if (epoch+1) % self.print_every == 0 or (epoch==0):
                print('{:5d} | {:10.5f} | {:10.5f} | {:10.5f} | {:10.5f} | {:10.5f} | {:10.5f} | {:>10} | {:10.1f}'.format(
                    epoch+1, val_stats['euler'], val_stats['L1'], val_stats['L2'], val_stats['L3'],
                    val_stats['psi_rmse'], val_stats['hess_rmse'], 
                    f"{val_stats['euler']/train_stats['euler']:.2f}", time.time()-t0))
        print('\nBest val metrics:')
        print({k: v for k, v in self.best.items()})

        # Save to DataFrame for later analysis
        self.history_df = pd.DataFrame(self.history)
        if self.metrics_save_path:
            self.history_df.to_csv(self.metrics_save_path, index=False)

    def _epoch_pass(self, loader, mode='train'):
        losses_euler, losses_L1, losses_L2, losses_L3, losses_symm = [], [], [], [], []
        rmse_Q, rmse_psi = [], []
        psi_r2s = []
        losses_psi, losses_psi_mse = [], []
        hess_eigenvals = []
        N = 0

        for A_batch, Q_batch in loader:
            A_batch, Q_batch = A_batch.to(self.device), Q_batch.to(self.device)
            s, w, eps, q, r = get_tensor_derivatives(A_batch)
            _, Q_hat_t, psi_t = process_ground_truth_Q(Q_batch, eps)
            a = A_batch / eps.unsqueeze(-1).unsqueeze(-1)
            s_norm = 0.5 * (a + a.transpose(1, 2))

            Q_hat_p, raw_Q_p = self.model_Q(s_norm, w)
            psi_p = self.model_psi(q, r)

            # Euler angle and component losses
            loss_euler, euler_dict = euler_angle_loss(Q_hat_p, Q_hat_t, s_norm)
            losses_euler.append(loss_euler.item())
            losses_L1.append(euler_dict.get('L1', 0))
            losses_L2.append(euler_dict.get('L2', 0))
            losses_L3.append(euler_dict.get('L3', 0))
            losses_symm.append(symmetry_loss(raw_Q_p).item())

            # RMSE and R2
            rmse_Q.append(torch.sqrt(F.mse_loss(Q_hat_p, Q_hat_t)).item())
            rmse_psi.append(torch.sqrt(F.mse_loss(psi_p, psi_t)).item())
            losses_psi.append(log_cosh_loss(psi_p, psi_t).item())
            losses_psi_mse.append(F.mse_loss(psi_p, psi_t).item())
            psi_r2s.append(self._r2(psi_p, psi_t))

            # Hessian (only for val, every few epochs)
            if mode == 'val' and (len(hess_eigenvals)==0 or N==0):
                try:
                    loss_fn_q = lambda pred, raw_pred, target, s: euler_angle_loss(pred, target, s)[0] + self.symm_loss_weight * symmetry_loss(raw_pred)
                    hess = self._get_hessian_max_eigenvalue(self.model_Q, loss_fn_q, (s_norm, w), (Q_hat_t, s_norm))
                except Exception:
                    hess = np.nan
                hess_eigenvals.append(hess)
            N += 1

            # Backprop only if train
            if mode == 'train':
                self.optimizer_Q.zero_grad(set_to_none=True)
                self.optimizer_psi.zero_grad(set_to_none=True)
                (loss_euler + self.symm_loss_weight * symmetry_loss(raw_Q_p)).backward(retain_graph=True)
                log_cosh_loss(psi_p, psi_t).backward()
                torch.nn.utils.clip_grad_norm_(self.model_Q.parameters(), self.grad_clip_value)
                torch.nn.utils.clip_grad_norm_(self.model_psi.parameters(), self.grad_clip_value)
                self.optimizer_Q.step()
                self.optimizer_psi.step()
                self.scheduler_Q.step()
                self.scheduler_psi.step()

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
            'hess_rmse': np.mean(hess_eigenvals) if hess_eigenvals else np.nan,
        }
        return res

    def _get_hessian_max_eigenvalue(self, model, loss_fn, inputs, targets, num_iterations=6):
        model.zero_grad()
        params = [p for p in model.parameters() if p.requires_grad]
        v = [torch.randn_like(p) for p in params]
        for _ in range(num_iterations):
            outputs, raw_outputs = model(*inputs)
            loss = loss_fn(outputs, raw_outputs, *targets)
            grad_params = torch.autograd.grad(loss, params, create_graph=True)
            hvp = torch.autograd.grad(grad_params, params, v, retain_graph=True)
            v_norm = torch.sqrt(sum(torch.sum(x * x) for x in hvp))
            v = [x / (v_norm + 1e-8) for x in hvp]
        outputs, raw_outputs = model(*inputs)
        loss = loss_fn(outputs, raw_outputs, *targets)
        grad_params = torch.autograd.grad(loss, params, create_graph=True)
        hvp = torch.autograd.grad(grad_params, params, v, retain_graph=True)
        eigenvalue = sum(torch.sum(x * y) for x, y in zip(hvp, v)).item()
        return eigenvalue

    def _r2(self, pred, true):
        pred, true = pred.flatten().detach().cpu().numpy(), true.flatten().detach().cpu().numpy()
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - np.mean(true))**2) + 1e-8
        return 1 - ss_res / ss_tot

    def plot_history(self, save_dir=None):
        df = pd.DataFrame(self.history)
        epochs = df['epoch']
        
        def add_best_ema(ax, train_arr, val_arr, label, train_c, val_c, ylbl=None):
            train_arr = np.asarray(train_arr)
            val_arr = np.asarray(val_arr)
            
            # Plot training metrics
            ax.plot(epochs, train_arr, train_c, label=f"Train {label}", lw=2, marker='o', ms=4)
            ax.plot(epochs, ema(train_arr, self.ema_alpha), train_c+'--', lw=2, label=f"Train {label} EMA")
            
            # Plot validation metrics
            ax.plot(epochs, val_arr, val_c, label=f"Val {label}", lw=2, marker='s', ms=4)
            ax.plot(epochs, ema(val_arr, self.ema_alpha), val_c+'--', lw=2, label=f"Val {label} EMA")
            
            # Add best validation point
            min_ix = np.argmin(val_arr)
            ax.axhline(np.min(val_arr), c=val_c, lw=1.2, ls=':', alpha=0.5)
            ax.plot(epochs[min_ix], val_arr[min_ix], marker='*', c=val_c, ms=14, mec='k', mew=1.5, label=f"Val {label} Best")
            
            if ylbl: ax.set_ylabel(ylbl)
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 11))
        
        # Euler angle and L1/L2/L3
        add_best_ema(axs[0,0], df['train_euler'], df['val_euler'], "Euler(Q)", 'C0', 'C1', "Euler Angle Loss")
        axs[0,0].plot(epochs, df['train_L1'], 'g-', lw=1.3, label="Train L1", alpha=0.7)
        axs[0,0].plot(epochs, df['val_L1'], 'g-', lw=1.3, label="Val L1")
        axs[0,0].plot(epochs, df['train_L2'], 'b-', lw=1.3, label="Train L2", alpha=0.7)
        axs[0,0].plot(epochs, df['val_L2'], 'b-', lw=1.3, label="Val L2")
        axs[0,0].plot(epochs, df['train_L3'], 'r-', lw=1.3, label="Train L3", alpha=0.7)
        axs[0,0].plot(epochs, df['val_L3'], 'r-', lw=1.3, label="Val L3")
        axs[0,0].legend()
        
        # Psi RMSE and R2
        add_best_ema(axs[0,1], df['train_psi_rmse'], df['val_psi_rmse'], "Psi RMSE", 'C2', 'C3', "Psi RMSE")
        axs[0,1].plot(epochs, df['train_psi_r2'], 'C4--', lw=1.2, label="Train Psi R²", alpha=0.7)
        axs[0,1].plot(epochs, df['val_psi_r2'], 'C5--', lw=1.2, label="Val Psi R²")
        axs[0,1].legend()
        
        # Q RMSE and Hessian
        add_best_ema(axs[1,0], df['train_Q_rmse'], df['val_Q_rmse'], "Q RMSE", 'C6', 'C7', "Q RMSE")
        axs[1,0].plot(epochs, df['train_hess_rmse'], 'C8-', lw=1.3, label="Train Hessian λ_max", alpha=0.7)
        axs[1,0].plot(epochs, df['val_hess_rmse'], 'C9-', lw=1.3, label="Val Hessian λ_max")
        axs[1,0].legend()
        
        # Symmetry loss, logcosh, mse
        add_best_ema(axs[1,1], df['train_symm'], df['val_symm'], "Symmetry", 'C10', 'C11', "Symmetry Loss")
        axs[1,1].plot(epochs, df['train_psi_logcosh'], 'C12-', lw=1.3, label="Train Psi LogCosh", alpha=0.7)
        axs[1,1].plot(epochs, df['val_psi_logcosh'], 'C13-', lw=1.3, label="Val Psi LogCosh")
        axs[1,1].plot(epochs, df['train_psi_mse'], 'C14-', lw=1.3, label="Train Psi MSE", alpha=0.7)
        axs[1,1].plot(epochs, df['val_psi_mse'], 'C15-', lw=1.3, label="Val Psi MSE")
        axs[1,1].legend()
        
        for ax in axs.flat:
            ax.set_xlabel('Epoch')
            ax.grid(True, ls=':', alpha=0.6)
        
        fig.suptitle("AIBM Training & Validation Metrics", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.97])
        
        if save_dir:
            plt.savefig(f"{save_dir}/training_metrics_full.png", dpi=300)
        plt.show()

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
      to match the paper’s notation and the training loss utilities.
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
        self.mQ = model_Q.to(self.device).eval()
        self.mP = model_psi.to(self.device).eval()
        self.dataset = dataset
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)

        if save_dir is None:
            save_dir = str(globals().get('RUN_DIR', Path('figs') / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._prepared = False

    def _prepare(self):
        if self._prepared:
            return

        A, Q = self.dataset.get_full_dataset()
        if self.max_samples is not None and len(A) > self.max_samples:
            idx = torch.from_numpy(self.rng.choice(len(A), size=self.max_samples, replace=False))
            A, Q = A[idx], Q[idx]

        A = A.to(self.device).float()
        Q = Q.to(self.device).float()

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

    def _hexbin_mean(self, ax, x, y, c, gridsize=220, vmin=None, vmax=None):
        return ax.hexbin(x, y, C=c, gridsize=gridsize, reduce_C_function=np.mean,
                         cmap='viridis', linewidths=0.1, vmin=vmin, vmax=vmax)

    def _pdf_curve(self, x, bins=140, clip=None):
        x = np.asarray(x).reshape(-1)
        if clip is not None:
            x = x[(x >= clip[0]) & (x <= clip[1])]
        if len(x) == 0:
            grid = np.linspace(0, 1, bins)
            return grid, np.zeros_like(grid)
        h, e = np.histogram(x, bins=bins, density=True)
        xc = 0.5 * (e[:-1] + e[1:])
        return xc, h

    # ------------------------------------------------------------------ #
    # 1) ψ(q,r) — separate DNS and AIBM (no overlay) + legends/colorbars
    # ------------------------------------------------------------------ #
    def fig_01_psi_qr(self):
        self._prepare()
        q, r = self.q_np, self.r_np
        vmin, vmax = 0.0, float(np.percentile(np.concatenate([self.psi_true_np, self.psi_pred_np]), 99.5))

        # DNS
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hb = self._hexbin_mean(ax, r, q, self.psi_true_np, vmin=vmin, vmax=vmax)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'w--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'w--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title(r'DNS: $\psi(q,r)$'); ax.grid(True, ls=':', alpha=0.45)
        cb = fig.colorbar(hb, ax=ax, pad=0.01); cb.set_label(r'$\langle \psi \rangle$')
        ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '01A_psi_qr_dns.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

        # AIBM
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        hb = self._hexbin_mean(ax, r, q, self.psi_pred_np, vmin=vmin, vmax=vmax)
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'w--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'w--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title(r'AIBM: $\psi(q,r)$'); ax.grid(True, ls=':', alpha=0.45)
        cb = fig.colorbar(hb, ax=ax, pad=0.01); cb.set_label(r'$\langle \psi \rangle$')
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
        kde = sns.kdeplot(x=r, y=q, fill=True, levels=40, thresh=1e-4, ax=ax, cmap='viridis')
        if getattr(kde, "collections", None):
            fig.colorbar(kde.collections[0], ax=ax, pad=0.01, label='PDF (a.u.)')
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'k--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'k--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title('DNS: joint PDF in $(q,r)$')
        ax.grid(True, ls=':', alpha=0.45); ax.legend(loc='upper left')
        fig.savefig(self.save_dir / '02A_qr_pdf_dns.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

        # AIBM (same (q,r) cloud; still shown separately)
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        kde = sns.kdeplot(x=r, y=q, fill=True, levels=40, thresh=1e-4, ax=ax, cmap='mako')
        if getattr(kde, "collections", None):
            fig.colorbar(kde.collections[0], ax=ax, pad=0.01, label='PDF (a.u.)')
        qv, rp, rm = self._vieillefosse(q.min(), q.max())
        ax.plot(rp, qv, 'k--', lw=1.1, label='Vieillefosse'); ax.plot(rm, qv, 'k--', lw=1.1)
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$q$'); ax.set_title('AIBM: joint PDF in $(q,r)$')
        ax.grid(True, ls=':', alpha=0.45); ax.legend(loc='upper left')
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
            axs[k].set_yscale('log')
        axs[0].legend()
        fig.savefig(self.save_dir / '03A_rotation_invariance_dns.png', dpi=self.DPI); plt.close(fig)

        # DNS vs AIBM overlay
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
        for k in range(3):
            sns.kdeplot(F_dns[:, k], ax=axs[k], bw_adjust=0.9, color="k", lw=1.8, label="DNS")
            sns.kdeplot(F_mod[:, k], ax=axs[k], bw_adjust=0.9, color="C1", lw=1.8, label="AIBM")
            axs[k].set_title(titles[k]); axs[k].set_xlabel("Ratio"); axs[k].grid(True, ls=':', alpha=0.45)
            axs[k].set_yscale('log')
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
                xs, ys = self._pdf_curve(cosabs(s_e[i], t_e[j]), bins=120, clip=(0, 1))
                ax = axs[i, j]
                ax.plot(xs, ys, 'k-', lw=2, label='DNS')
                ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
                ax.set_title(labels[i][j]); ax.grid(True, ls=':', alpha=0.5)
        axs[0, 0].legend(loc='upper right')
        fig.savefig(self.save_dir / '04A_sQ_align_dns.png', dpi=self.DPI); plt.close(fig)

        # DNS vs AIBM
        fig, axs = plt.subplots(3, 3, figsize=(12.6, 10.6), constrained_layout=True)
        for i in range(3):
            for j in range(3):
                ax = axs[i, j]
                xs, ys = self._pdf_curve(cosabs(s_e[i], t_e[j]), bins=120, clip=(0, 1))
                ax.plot(xs, ys, 'k-', lw=2, label='DNS')
                xs2, ys2 = self._pdf_curve(cosabs(s_e[i], p_e[j]), bins=120, clip=(0, 1))
                ax.plot(xs2, ys2, 'C1--', lw=2, label='AIBM')
                ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
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
            xs, ys = self._pdf_curve(cosabs(omg, t_e[j]), bins=120, clip=(0, 1))
            ax = axs[j]; ax.plot(xs, ys, 'k-', lw=2, label='DNS')
            ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
            ax.set_title(labels[j]); ax.grid(True, ls=':', alpha=0.5)
        axs[0].legend(loc='upper right')
        fig.savefig(self.save_dir / '05A_wQ_align_dns.png', dpi=self.DPI); plt.close(fig)

        # DNS vs AIBM
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
        for j in range(3):
            ax = axs[j]
            xs, ys = self._pdf_curve(cosabs(omg, t_e[j]), bins=120, clip=(0, 1))
            ax.plot(xs, ys, 'k-', lw=2, label='DNS')
            xs2, ys2 = self._pdf_curve(cosabs(omg, p_e[j]), bins=120, clip=(0, 1))
            ax.plot(xs2, ys2, 'C1--', lw=2, label='AIBM')
            ax.set_xlim(0, 1); ax.set_xlabel('|cos θ|'); ax.set_ylabel('PDF')
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
                                 clip=(np.percentile(psi_t, 0.1), np.percentile(psi_t, 99.9)))
        ax.plot(xs, ys, 'k-', lw=2, label='DNS')
        ax.set_xlabel(r'$\psi$'); ax.set_ylabel('PDF'); ax.grid(True, ls=':', alpha=0.5)
        ax.set_title(r'DNS: marginal PDF of $\psi$'); ax.legend()
        fig.savefig(self.save_dir / '06A_psi_pdf_dns.png', dpi=self.DPI); plt.close(fig)

        # Overlay + RMSE vertical line
        fig, ax = plt.subplots(figsize=(6.9, 4.8))
        xs, ys = self._pdf_curve(psi_t, bins=160,
                                 clip=(np.percentile(psi_t, 0.1), np.percentile(psi_t, 99.9)))
        ax.plot(xs, ys, 'k-', lw=2, label='DNS')
        xs2, ys2 = self._pdf_curve(psi_p, bins=160,
                                   clip=(np.percentile(psi_p, 0.1), np.percentile(psi_p, 99.9)))
        ax.plot(xs2, ys2, 'C1--', lw=2, label='AIBM')
        ax.axvline(rmse, color='C3', lw=1.8, ls='-.', label=f'RMSE = {rmse:.3f}')
        ax.set_xlabel(r'$\psi$'); ax.set_ylabel('PDF'); ax.grid(True, ls=':', alpha=0.5)
        ax.set_title(r'DNS vs AIBM: marginal PDF of $\psi$'); ax.legend()
        fig.savefig(self.save_dir / '06B_psi_pdf_overlay_rmse.png', dpi=self.DPI); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 7) φ vs qε² — NO EMA; DNS-only and DNS/AIBM each with full & zoom
    # ------------------------------------------------------------------ #
    # ---------- inside class AIBMVisualizer ----------

    @staticmethod
    def _std_2pass(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size
        if n <= 1: return 0.0
        mu = float(np.mean(x, dtype=np.float64))
        ssd = float(np.sum((x - mu)**2, dtype=np.float64))
        return float(np.sqrt(ssd / max(n - 1, 1)))

    @staticmethod
    def _binned_mean_precise(x: np.ndarray, y: np.ndarray, edges: np.ndarray,
                            min_count: int = 20, interpolate_gaps: bool = True):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        m = len(edges) - 1
        xc = 0.5 * (edges[:-1] + edges[1:])
        idx = np.searchsorted(edges, x, side='right') - 1
        ok = (idx >= 0) & (idx < m)
        if not np.any(ok):
            return xc, np.full(m, np.nan), np.zeros(m, int)
        idx = idx[ok]; y = y[ok]
        cnt = np.bincount(idx, minlength=m).astype(int)
        s   = np.bincount(idx, weights=y, minlength=m).astype(np.float64)
        mean = np.full(m, np.nan, float)
        nz = cnt > 0
        mean[nz] = s[nz] / cnt[nz]
        mean[cnt < min_count] = np.nan
        if interpolate_gaps and np.isnan(mean).any():
            good = ~np.isnan(mean)
            if np.any(good):
                mean = np.interp(np.arange(m, dtype=float),
                                np.flatnonzero(good).astype(float),
                                mean[good])
        return xc, mean, cnt

    def _phi_profile_precise(self, X: np.ndarray, Qsq: np.ndarray,
                            nbins: int, xrng: tuple[float,float] | None,
                            min_count: int = 20, interpolate_gaps: bool = True):
        X   = np.asarray(X,   dtype=np.float64).ravel()
        Qsq = np.asarray(Qsq, dtype=np.float64).ravel()
        sigX = self._std_2pass(X)  + 1e-15
        sigQ = self._std_2pass(Qsq)+ 1e-15
        if xrng is None:
            xmin, xmax = np.percentile(X, [0.5, 99.5])
        else:
            xmin, xmax = xrng
        edges = np.linspace(xmin, xmax, nbins + 1, dtype=np.float64)
        xc, cond, _ = self._binned_mean_precise(X, Qsq, edges,
                                                min_count=min_count,
                                                interpolate_gaps=interpolate_gaps)
        phi = cond * (sigX / sigQ)           # Eq. (31)
        return xc, phi

    def fig_07_phi_qe2(self, nbins: int = 160, min_count: int = 20):
        """
        φ(x) = E[||Q||² | x] * (σ_{qε²} / σ_{||Q||²}), x = q ε².
        Left: full range (native x). Right: zoom with standardized abscissa Xσ = x/σ_x (±1σ).
        """
        self._prepare()
        X   = self.X_qe2_np                # q ε²  (native)
        Y_D = self.Y_true_np               # ||Q||² DNS
        Y_M = self.Y_pred_np               # ||Q||² AIBM

        # ---- full range (native) ----
        xcD_full, phiD_full = self._phi_profile_precise(X, Y_D, nbins, xrng=None,
                                                        min_count=min_count, interpolate_gaps=False)
        xcM_full, phiM_full = self._phi_profile_precise(X, Y_M, nbins, xrng=None,
                                                        min_count=min_count, interpolate_gaps=False)

        # ---- zoom (standardized abscissa) ----
        sigX = self._std_2pass(X) + 1e-15
        Xs   = X / sigX                      # Xσ has std≈1 ⇒ [-1,1] is populated
        edges_zoom = np.linspace(-1.0, +1.0, nbins + 1, dtype=np.float64)

        xz, phiD_zoom, _ = self._binned_mean_precise(Xs, 
                            Y_D * ( (self._std_2pass(X)+1e-15) / (self._std_2pass(Y_D)+1e-15) ),
                            edges_zoom, min_count=min_count, interpolate_gaps=True)
        _,  phiM_zoom, _ = self._binned_mean_precise(Xs, 
                            Y_M * ( (self._std_2pass(X)+1e-15) / (self._std_2pass(Y_M)+1e-15) ),
                            edges_zoom, min_count=min_count, interpolate_gaps=True)

        # ---- (A) DNS only ----
        fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
        axs[0].plot(xcD_full, phiD_full, 'k-', lw=2, label='DNS')
        axs[0].set_xlabel(r'$q\,\varepsilon^{2}$'); axs[0].set_ylabel(r'$\phi(x)$')
        axs[0].set_title('DNS: full range (native)'); axs[0].grid(True, ls=':', alpha=0.5); axs[0].legend()

        axs[1].plot(xz, phiD_zoom, 'k-', lw=2, label='DNS')
        axs[1].set_xlim(-1.0, +1.0); axs[1].set_ylim(0.00, 0.25)
        axs[1].set_xlabel(r'$q\,\varepsilon^{2}/\sigma_{q\varepsilon^{2}}$'); axs[1].set_ylabel(r'$\phi(x)$')
        axs[1].set_title('DNS: zoom (standardized abscissa)'); axs[1].grid(True, ls=':', alpha=0.5); axs[1].legend()
        fig.savefig(self.save_dir / '07A_phi_qe2_dns_full_and_zoom.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

        # ---- (B) DNS vs AIBM ----
        fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
        axs[0].plot(xcD_full, phiD_full, 'k-', lw=2, label='DNS')
        axs[0].plot(xcM_full, phiM_full, 'C1--', lw=2, label='AIBM')
        axs[0].set_xlabel(r'$q\,\varepsilon^{2}$'); axs[0].set_ylabel(r'$\phi(x)$')
        axs[0].set_title('DNS vs AIBM: full range (native)'); axs[0].grid(True, ls=':', alpha=0.5); axs[0].legend()

        axs[1].plot(xz, phiD_zoom, 'k-', lw=2, label='DNS')
        axs[1].plot(xz, phiM_zoom, 'C1--', lw=2, label='AIBM')
        axs[1].set_xlim(-1.0, +1.0); axs[1].set_ylim(0.00, 0.25)
        axs[1].set_xlabel(r'$q\,\varepsilon^{2}/\sigma_{q\varepsilon^{2}}$'); axs[1].set_ylabel(r'$\phi(x)$')
        axs[1].set_title('DNS vs AIBM: zoom (standardized abscissa)'); axs[1].grid(True, ls=':', alpha=0.5); axs[1].legend()
        fig.savefig(self.save_dir / '07B_phi_qe2_overlay_full_and_zoom.png', dpi=self.DPI, bbox_inches='tight'); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Orchestrator
    # ------------------------------------------------------------------ #
    def plot_all(self):
        self.fig_01_psi_qr()                     # 01A, 01B (legends + colorbars)
        self.fig_02_qr_pdf()                     # 02A, 02B (colorbars + legends)
        self.fig_03_rotation_invariance(30.0)    # 03A, 03B
        self.fig_04_s_vs_Q()                     # 04A, 04B (ASC order; requested layout)
        self.fig_05_w_vs_Q()                     # 05A, 05B (ASC order; requested layout)
        self.fig_06_psi_pdf()                    # 06A, 06B (RMSE line)
        self.fig_07_phi_qe2()                    # 07A, 07B (full & zoom)


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
    args = parser.parse_args()

    # -------- Set up RUN_DIR --------
    if args.run_dir:
        RUN_DIR = pathlib.Path(args.run_dir)
    else:
        # Find the latest run_* directory by name (sort lexically)
        run_dirs = sorted([d for d in pathlib.Path('figs').glob('run_*') if d.is_dir()])
        if not run_dirs:
            raise RuntimeError("No previous run directories found in 'figs/'.")
        RUN_DIR = run_dirs[-1]
    print(f"Using RUN_DIR: {RUN_DIR}")

    # -------- Data Load (same as before) --------
    try:
        scipy.io.loadmat('velGrad.mat'); scipy.io.loadmat('PH.mat')
    except FileNotFoundError:
        print("Creating dummy .mat files for demonstration...")
        N = 100000; vel_grad_dummy = np.random.randn(9, N).astype(np.float32); ph_dummy = np.random.randn(N, 9).astype(np.float32)
        scipy.io.savemat('velGrad.mat', {'velGrad': vel_grad_dummy}); scipy.io.savemat('PH.mat', {'PH': ph_dummy})

    full_dataset = MatlabDataset('velGrad.mat', 'PH.mat')
    train_size = int(0.8 * len(full_dataset)); val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16384, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=65536, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    model_Q = TBNN_Q_direction()
    model_psi = FCNN_psi_magnitude()

    trainer = AIBMTrainer(model_Q, model_psi, train_loader, val_loader, epochs=1000, hessian_freq=10, learning_rate_Q=4e-3, learning_rate_psi=20e-3, print_every=1)

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
        print("No checkpoint restored, training from scratch.")
        RUN_DIR = pathlib.Path('figs') / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        print("Directory made:",RUN_DIR)

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
