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

def vieillefosse_curve(n_pts: int = 300):
    qv = np.linspace(-1, 1, n_pts)
    rv = (2.0 / 3.0) * np.sqrt(3) * np.abs(qv) ** 1.5
    msk = qv <= 0
    return qv[msk], +rv[msk], -rv[msk]

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
        def add_best_ema(ax, arr, label, c, ylbl=None):
            arr = np.asarray(arr)
            ax.plot(epochs, arr, c, label=label, lw=2, marker='o', ms=4)
            ax.plot(epochs, ema(arr, self.ema_alpha), c+'--', lw=2, label=label+" EMA")
            min_ix = np.argmin(arr)
            ax.axhline(np.min(arr), c=c, lw=1.2, ls=':', alpha=0.5)
            ax.plot(epochs[min_ix], arr[min_ix], marker='*', c=c, ms=14, mec='k', mew=1.5, label=label+' Best')
            if ylbl: ax.set_ylabel(ylbl)
        fig, axs = plt.subplots(2, 2, figsize=(16, 11))
        # Euler angle and L1/L2/L3
        add_best_ema(axs[0,0], df['val_euler'], "Val Euler(Q)", 'C0', "Euler Angle Loss")
        axs[0,0].plot(epochs, df['val_L1'], 'g-', lw=1.3, label="L1")
        axs[0,0].plot(epochs, df['val_L2'], 'b-', lw=1.3, label="L2")
        axs[0,0].plot(epochs, df['val_L3'], 'r-', lw=1.3, label="L3")
        axs[0,0].legend()
        # Psi RMSE and R2
        add_best_ema(axs[0,1], df['val_psi_rmse'], "Val Psi RMSE", 'C1', "Psi RMSE")
        axs[0,1].plot(epochs, df['val_psi_r2'], 'C3--', lw=1.2, label="Psi R²")
        axs[0,1].legend()
        # Q RMSE and Hessian
        add_best_ema(axs[1,0], df['val_Q_rmse'], "Q RMSE", 'C4', "Q RMSE")
        axs[1,0].plot(epochs, df['val_hess_rmse'], 'C5-', lw=1.3, label="Hessian λ_max")
        axs[1,0].legend()
        # Symmetry loss, logcosh, mse
        add_best_ema(axs[1,1], df['val_symm'], "Val Symmetry", 'C2', "Symmetry Loss")
        axs[1,1].plot(epochs, df['val_psi_logcosh'], 'C6-', lw=1.3, label="Psi LogCosh")
        axs[1,1].plot(epochs, df['val_psi_mse'], 'C7-', lw=1.3, label="Psi MSE")
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
    End-to-end diagnostic plots for the AIBM pressure-Hessian model.
    Call `plot_all()` after instantiation.
    """
    def __init__(self,
                model_Q: torch.nn.Module,
                model_psi: torch.nn.Module,
                data_generator,
                run_dir: str | Path = "figs/run_vis"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ––––– DNS tensors –––––
        self.A_full, self.Q_full = data_generator.get_full_dataset()  # torch tensors
        self.A_full = self.A_full.to(self.device)
        self.Q_full = self.Q_full.to(self.device)

        # Velocity-gradient decomposition
        self.s, self.w, self.eps, self.q, self.r = get_tensor_derivatives(self.A_full)
        self.Qp, self.Qhp, self.psi_true = process_ground_truth_Q(self.Q_full, self.eps)

        # Normalise A for model input
        a = self.A_full / self.eps.unsqueeze(-1).unsqueeze(-1)
        self.s_norm = 0.5 * (a + a.transpose(1, 2))

        # ––––– Model predictions –––––
        self.model_Q = model_Q.to(self.device).eval()
        self.model_psi = model_psi.to(self.device).eval()
        with torch.no_grad():
            self.Qhp_pred, _ = self.model_Q(self.s_norm, self.w)
            self.psi_pred = self.model_psi(self.q, self.r)

        # Move to cpu-numpy for plotting
        self._to_numpy()

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────
    def plot_all(self):
        """Create every figure (7 total)."""
        # self.plot_psi_contour_on_qr()
        # self.plot_qr_joint_pdf()
        self.plot_invariant_ratios()
        # self.plot_eigenvector_alignment_s_Q()
        # self.plot_eigenvector_alignment_vorticity_Q()
        # self.plot_psi_pdf()
        # self.plot_phi_vs_q_epsilon_sq()

    # ──────────────────────────────────────────────────────────────────────
    #  Individual panels
    # ──────────────────────────────────────────────────────────────────────
    # 1 – ψ(q,r) contour with Vieillefosse tail
    def plot_psi_contour_on_qr(self, show: bool = False):
        
        print("Generating psi contours of < psi | q,r >")

        fig, ax = plt.subplots(figsize=(7, 5))
        h = hexbin_with_mean(
            ax, self.r_np, self.q_np,
            np.clip(self.psi_true_np, 0, 2),
            vmin=0, vmax=2
        )
        cb = fig.colorbar(h, ax=ax)
        cb.set_label(r"$\langle \psi \mid q,r \rangle$")
        self._format_qr_axes(ax)
        # Vieillefosse
        qv, rv_p, rv_m = vieillefosse_curve()
        ax.plot(rv_p, qv, 'r--', lw=1.5)
        ax.plot(rv_m, qv, 'r--', lw=1.5)
        fig.savefig(self.run_dir / "fig_psi_contour_qr.png")
        if show: plt.show()
        plt.close(fig)

    # 2 – joint PDF of (q,r)
    def plot_qr_joint_pdf(self, show: bool = False):

        print("Generating joint-PDf of (q,r)")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(
            x=self.r_np, y=self.q_np, fill=True,
            cmap="mako", levels=100, bw_adjust=0.3,
            clip=[[-1, 1], [-1, 1]], ax=ax
        )
        self._format_qr_axes(ax)
        qv, rv_p, rv_m = vieillefosse_curve()
        ax.plot(rv_p, qv, 'r--', lw=1.3)
        ax.plot(rv_m, qv, 'r--', lw=1.3)
        fig.savefig(self.run_dir / "fig_qr_pdf.png")
        if show: plt.show()
        plt.close(fig)

    # 3 – Invariant–ratio PDFs (DNS & model)
    def plot_invariant_ratios(self, show: bool = False):
        """
        PDFs of F1, F2, F3 defined in Eq. (37) of the paper.
        For a rotation-invariant model the distributions collapse to δ(Fk-1).
        """

        print("Generating PDFs of invariant ratios F1,F2,F3")
 
        # 30° rotation about z
        R = self._rotation_matrix_z(np.pi / 6).to(self.device)

        # Helper: invariants from eigenvalues
        def invariants(eig):                           # eig: (N,3)
            I1 = eig.sum(1)
            I2 = eig[:,0]*eig[:,1] + eig[:,1]*eig[:,2] + eig[:,0]*eig[:,2]
            I3 = eig.prod(1)
            return I1, I2, I3

        # ─── DNS ──────────────────────────────────────────────────────────
        Q_dns      = self.Qp                                # (N,3,3)
        Q_dns_rot  = torch.einsum('ik,bkl,jl->bij', R, Q_dns, R)
        eig_dns,   eig_dns_r   = map(np.linalg.eigvalsh,
                                     (Q_dns.cpu().numpy(),
                                      Q_dns_rot.cpu().numpy()))
        I1, I2, I3       = invariants(eig_dns)
        I1r, I2r, I3r    = invariants(eig_dns_r)
        F1_dns = I1r/I1;  F2_dns = I2r/I2;  F3_dns = I3r/I3

        # ─── AIBM ────────────────────────────────────────────────────────
        # original prediction (dimensional)
        Q_pred_dim = self.Qhp_pred * self.psi_pred[:,None,None]
        # rotated inputs
        A_rot      = torch.einsum('ik,bkl,jl->bij', R, self.A_full, R)
        s_rot, w_rot, eps_rot, q_rot, r_rot = get_tensor_derivatives(A_rot)
        a_rot   = A_rot / eps_rot[:,None,None]
        s_rot_n = 0.5*(a_rot + a_rot.transpose(1,2))
        with torch.no_grad():
            Qhat_rot,_ = self.model_Q(s_rot_n, w_rot)
            psi_rot    = self.model_psi(q_rot, r_rot)
        Q_pred_dim_rot = Qhat_rot * psi_rot[:,None,None]

        eig_pred, eig_pred_r = map(np.linalg.eigvalsh,
                                   (Q_pred_dim.cpu().numpy(),
                                    Q_pred_dim_rot.cpu().numpy()))
        I1p, I2p, I3p    = invariants(eig_pred)
        I1pr, I2pr, I3pr = invariants(eig_pred_r)
        F1_pred = I1pr/I1p;  F2_pred = I2pr/I2p;  F3_pred = I3pr/I3p

        # ─── Plot ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        for k,(Fd,Fp,title) in enumerate(zip(
                [F1_dns, F2_dns, F3_dns],
                [F1_pred, F2_pred, F3_pred],
                [r"$F_1 = I_1'/I_1$", r"$F_2 = I_2'/I_2$", r"$F_3 = I_3'/I_3$"])):
            ax = axes[k]
            sns.kdeplot(Fd, ax=ax, label="DNS",  lw=1)
            sns.kdeplot(Fp, ax=ax, label="AIBM", lw=1.25, ls='--')
            # ax.axvline(1, c='grey', ls=':')
            ax.set_xlim(([(-4, 4), (0.99, 1.01), (0.85, 1.05)])[k])      # loose bounds; adjust if needed
            if k==2: ax.set_yscale('log')
            ax.set_xlabel(title)
            ax.set_ylabel("PDF")
            ax.grid(ls=':', alpha=0.7)
            if k==2: ax.legend()
        fig.suptitle("Rotation-invariance check via invariant ratios", fontsize=12)
        fig.savefig(self.run_dir / "fig_invariant_ratios.png", dpi=1200)
        if show: plt.show()
        plt.close(fig)


    # 4 – |cos (s ↔ Q′)| PDFs (9 panels)
    def plot_eigenvector_alignment_s_Q(self, show: bool = False):

        print("Generating PDFs of cosine-angles dotted of s with Q'")
        _, vec_s = eig_desc(self.s_np)
        _, vec_q_dns = eig_desc(self.Qhp_np)
        _, vec_q_pred = eig_desc(self.Qhp_pred_np)

        labels_s = [r"$\hat{e}_{\gamma_s}$",
                    r"$\hat{e}_{\beta_s}$",
                    r"$\hat{e}_{\alpha_s}$"]
        labels_p = [r"$\hat{e}_{\gamma_{Q'}}$",
                    r"$\hat{e}_{\beta_{Q'}}$",
                    r"$\hat{e}_{\alpha_{Q'}}$"]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        for i in range(3):
            for j in range(3):
                c_dns  = self._abs_cos(vec_s[:, :, i], vec_q_dns[:, :, j])
                c_pred = self._abs_cos(vec_s[:, :, i], vec_q_pred[:, :, j])
                ax = axes[i, j]
                sns.kdeplot(c_dns,  ax=ax, label="DNS",   lw=2,
                            bw_adjust=0.4, clip=(0, 1))
                sns.kdeplot(c_pred, ax=ax, label="AIBM", linestyle='--', lw=2,
                            bw_adjust=0.4, clip=(0, 1))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 2)
                ax.set_title(f"{labels_s[i]} vs {labels_p[j]}")
                ax.axvline(1, ls=':', c='grey')
                if i == 0 and j == 2:
                    ax.legend()
                ax.grid(ls=':', alpha=0.7)
        fig.suptitle(r"PDFs of $|\,\cos(\theta_{s,Q'})|$", fontsize=16)
        fig.savefig(self.run_dir / "fig_cos_s_Q.png")
        if show: plt.show()
        plt.close(fig)

    # 5 – |cos (ω ↔ Q′)| PDFs (3 panels)
    def plot_eigenvector_alignment_vorticity_Q(self, show: bool = False):

        print("Generating PDFs of cosine-angles dotted of Q' with vorticity")
        vort = self._vorticity_vector(self.w_np)
        _, vec_q_dns = eig_desc(self.Qhp_np)
        _, vec_q_pred = eig_desc(self.Qhp_pred_np)
        labels = [r"$\hat{e}_{\gamma_{Q'}}$",
                r"$\hat{e}_{\beta_{Q'}}$",
                r"$\hat{e}_{\alpha_{Q'}}$"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i in range(3):
            c_dns  = self._abs_cos(vort, vec_q_dns[:, :, i])
            c_pred = self._abs_cos(vort, vec_q_pred[:, :, i])
            ax = axes[i]
            sns.kdeplot(c_dns,  ax=ax, lw=2, clip=(0, 1))
            sns.kdeplot(c_pred, ax=ax, lw=2, linestyle='--', clip=(0, 1))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2)
            ax.set_xlabel("Absolute cosine")
            ax.set_title(f"ω · {labels[i]}")
            ax.axvline(1, ls=':', c='grey')
            if i == 2:
                ax.legend(["DNS", "AIBM"])
            ax.grid(ls=':', alpha=0.7)
        fig.suptitle(r"PDFs of $|\,\cos(\theta_{\omega,Q'})|$", fontsize=16)
        fig.savefig(self.run_dir / "fig_cos_w_Q.png")
        if show: plt.show()
        plt.close(fig)

    # 6 – ψ PDF
    def plot_psi_pdf(self, show: bool = False):

        print("Generating PDF of psi's distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(self.psi_true_np, ax=ax, lw=2,
                    bw_adjust=0.4, clip=(0, 8), label="DNS")
        sns.kdeplot(self.psi_pred_np, ax=ax, lw=2, linestyle='--',
                    bw_adjust=0.4, clip=(0, 8), label="AIBM")
        ax.set_xlim(0, 8)
        ax.set_xlabel(r"$\psi$")
        ax.set_ylabel("PDF")
        ax.set_title(r"PDF of $\psi$")
        ax.legend()
        ax.grid(ls=':', alpha=0.7)
        fig.savefig(self.run_dir / "fig_psi_pdf.png")
        if show: plt.show()
        plt.close(fig)

    # 7 – φ(qε²) relationship
    def plot_phi_vs_q_epsilon_sq(self, show: bool = False):

        print("Generating the phi v/s q-epsilone^2 graph")
        qe2 = self.q_np * self.eps_np ** 2
        msk = np.logical_and(qe2 >= -1, qe2 <= 1)

        Qmn_dns = np.sum(self.Qp_np[msk] ** 2, axis=(1, 2))
        Qmn_pred = np.sum(
            (self.Qhp_pred_np[msk] * self.psi_pred_np[msk, None, None]) ** 2,
            axis=(1, 2)
        )
        σ = np.std(Qmn_dns) + 1e-12
        φ_dns, φ_pred = Qmn_dns / σ, Qmn_pred / σ
        bins = np.linspace(-1, 1, 80)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        φ_dns_binned = self._bin_average(qe2[msk], φ_dns, bins)
        φ_pred_binned = self._bin_average(qe2[msk], φ_pred, bins)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(bin_centres, φ_dns_binned, 'v-', label="DNS", ms=7)
        ax.plot(bin_centres, φ_pred_binned, '*-', label="AIBM", ms=8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"$q\,\varepsilon^{2}$")
        ax.set_ylabel(r"$\phi$")
        ax.set_title(r"$\phi$ vs $q\varepsilon^{2}$")
        ax.axhline(0, ls=':', c='k')
        ax.grid(ls=':', alpha=0.7)
        ax.legend()
        fig.savefig(self.run_dir / "fig_phi_qe2.png")
        if show: plt.show()
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _to_numpy(self):
        """Detach & move every tensor needed for plotting to CPU-numpy."""
        self.q_np   = self.q.cpu().numpy()
        self.r_np   = self.r.cpu().numpy()
        self.eps_np = self.eps.cpu().numpy()
        self.s_np   = self.s_norm.cpu().numpy()
        self.w_np   = self.w.cpu().numpy()
        self.Qp_np  = self.Qp.cpu().numpy()
        self.Qhp_np = self.Qhp.cpu().numpy()
        self.Qhp_pred_np = self.Qhp_pred.cpu().numpy()
        self.psi_true_np = self.psi_true.cpu().numpy()
        self.psi_pred_np = self.psi_pred.cpu().numpy()

    @staticmethod
    def _rotation_matrix_z(theta: float):
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([[c, -s, 0],
                            [s,  c, 0],
                            [0,  0, 1]], dtype=torch.float32)

    @staticmethod
    def _abs_cos(a: np.ndarray, b: np.ndarray):
        """|cosθ| for two vector fields (N,3)."""
        dot = np.sum(a * b, axis=1)
        return np.clip(np.abs(dot), 0, 1)

    @staticmethod
    def _vorticity_vector(w: np.ndarray):
        vort = np.empty((w.shape[0], 3))
        vort[:, 0] = w[:, 2, 1] - w[:, 1, 2]
        vort[:, 1] = w[:, 0, 2] - w[:, 2, 0]
        vort[:, 2] = w[:, 1, 0] - w[:, 0, 1]
        norm = np.linalg.norm(vort, axis=1, keepdims=True) + 1e-12
        return vort / norm

    @staticmethod
    def _eigen_ratio(t_orig: torch.Tensor, t_rot: torch.Tensor):
        eig_o = np.linalg.eigvalsh(t_orig.detach().cpu().numpy())
        eig_r = np.linalg.eigvalsh(t_rot.detach().cpu().numpy())
        return (eig_r / (eig_o + 1e-8)).ravel()

    @staticmethod
    def _bin_average(x, y, bins):
        idx = np.digitize(x, bins) - 1
        out = np.full(len(bins) - 1, np.nan)
        for i in range(len(out)):
            sel = idx == i
            if sel.any():
                out[i] = y[sel].mean()
        return out

    @staticmethod
    def _format_qr_axes(ax):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("r")
        ax.set_ylabel("q")
        ax.grid(ls=':', alpha=0.7)

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
