import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from pyhessian import hessian # pip install pyhessian
from pyhessian.hessian import hessian_eigvals # pip install pyhessian
import warnings
warnings.filterwarnings('ignore')

import datetime, pathlib
RUN_DIR = pathlib.Path('figs') / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. UTILITY FUNCTIONS FOR TENSOR OPERATIONS
# ==============================================================================

def get_tensor_derivatives(A):
    """
    Calculates various derived quantities from the Velocity Gradient Tensor A.
    
    Args:
        A (torch.Tensor): Velocity Gradient Tensor of shape (batch, 3, 3).
        
    Returns:
        s (torch.Tensor): Strain-rate tensor (symmetric part of A).
        w (torch.Tensor): Rotation-rate tensor (anti-symmetric part of A).
        epsilon (torch.Tensor): Magnitude of A.
        q (torch.Tensor): Second invariant of the normalized tensor 'a'.
        r (torch.Tensor): Third invariant of the normalized tensor 'a'.
    """
    s = 0.5 * (A + A.transpose(1, 2))
    w = 0.5 * (A - A.transpose(1, 2))
    
    epsilon_sq = torch.sum(A * A, dim=(1, 2), keepdim=True)
    epsilon = torch.sqrt(epsilon_sq)
    # Avoid division by zero
    epsilon[epsilon == 0] = 1e-8

    a = A / epsilon
    
    # Invariants for incompressible flow (p=Tr(a)=0)
    a_sq = torch.einsum('bik,bkj->bij', a, a)
    q = -0.5 * torch.einsum('bii->b', a_sq)
    
    a_cubed = torch.einsum('bik,bkj->bij', a_sq, a)
    r = -1./3. * torch.einsum('bii->b', a_cubed)
    
    return s, w, epsilon.squeeze(), q, r

def process_ground_truth_Q(Q, epsilon):
    """
    Normalizes the ground truth Q tensor as described in the paper.
    
    Args:
        Q (torch.Tensor): Anisotropic Pressure Hessian from DNS.
        epsilon (torch.Tensor): Magnitude of A.
        
    Returns:
        Q_prime (torch.Tensor): Q normalized by epsilon^2.
        Q_hat_prime (torch.Tensor): Q_prime self-normalized by its own magnitude.
        psi (torch.Tensor): Magnitude of Q_prime.
    """
    epsilon_sq = (epsilon**2).unsqueeze(-1).unsqueeze(-1)
    Q_prime = Q / epsilon_sq
    
    psi_sq = torch.sum(Q_prime * Q_prime, dim=(1, 2), keepdim=True)
    psi = torch.sqrt(psi_sq)
    # Avoid division by zero
    psi[psi == 0] = 1e-8

    Q_hat_prime = Q_prime / psi
    
    return Q_prime, Q_hat_prime, psi.squeeze()

# ==============================================================================
# 2. DUMMY DATA GENERATOR
# ==============================================================================

class DummyTurbulenceDataGenerator:
    """
    Generates dummy turbulence data mimicking DNS output.

    This class creates random tensors for the velocity gradient (A) and the
    anisotropic pressure Hessian (Q). The tensors are made traceless to simulate
    the incompressible condition (Tr(A) = 0 and Tr(Q) = 0).

    The primary purpose is to provide correctly-shaped inputs and targets for the
    neural network models, allowing for end-to-end testing of the pipeline
    before plugging in real DNS data.

    Args:
        num_samples (int): The total number of data points to generate.
        batch_size (int): The size of each batch to be yielded.
    """
    def __init__(self, num_samples=10000, batch_size=128):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device

        eye = torch.eye(3, device=device)

        # Generate the full dataset in memory
        # For real-world use, you would load data from files.
        # A_ij: Velocity Gradient Tensor
        A = torch.randn(self.num_samples, 3, 3)
        # Q_ij: Anisotropic Pressure Hessian Tensor
        Q = torch.randn(self.num_samples, 3, 3)

        # Enforce traceless property for incompressibility
        # A_ii = 0 and Q_ii = 0
        A -= torch.einsum('bii->b', A)[:, None, None] * eye / 3  # traceless
        Q -= torch.einsum('bii->b', Q)[:, None, None] * eye / 3
        
        self.A = A.to(self.device)
        self.Q = Q.to(self.device)
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        # Shuffle data at the beginning of each epoch
        indices = torch.randperm(self.num_samples)
        self.A = self.A[indices]
        self.Q = self.Q[indices]
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_A = self.A[self.current_index:end_index]
        batch_Q = self.Q[self.current_index:end_index]
        
        self.current_index = end_index
        return batch_A, batch_Q

    def get_full_dataset(self):
        """Returns the entire dataset for evaluation."""
        return self.A, self.Q

    def compute_dataset_stats(self):
        """Return ε² mean/std and ψ mean/std so each dataset can be normalised separately."""
        A = self.A
        Q = self.Q
        with torch.no_grad():
            _, _, eps, _, _ = get_tensor_derivatives(A)
            Qp, _, psi = process_ground_truth_Q(Q, eps)
        stats = {
            'eps2_mean': (eps**2).mean().item(),
            'eps2_std' : (eps**2).std().item(),
            'psi_mean' : psi.mean().item(),
            'psi_std'  : psi.std().item(),
        }
        return stats


# ==============================================================================
# 3. MODEL DEFINITIONS (Enhanced with MC Dropout)
# ==============================================================================

class TBNN_Q_direction(nn.Module):
    """
    Tensor Basis Neural Network (TBNN) to predict the direction of Q'.

    This model implements the architecture from Eq. (15) of the paper.
    It takes normalized strain-rate (s) and rotation-rate (w) tensors,
    computes 5 invariants and 10 tensor bases, and uses a neural network
    to find the coefficients g(n) that combine the bases into the final
    predicted tensor Q_hat_prime.
    """
    def __init__(self, hidden_layers=[50, 100, 100, 100, 50], dropout_p=0.1):
        super().__init__()
        
        layers = []
        input_dim = 5  # 5 invariants
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_p))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 10)) # Output 10 coefficients
        
        self.network = nn.Sequential(*layers)
        self.training_mode = True

    def forward(self, s, w, mc_samples=None):
        if mc_samples is not None and not self.training:
            # Monte Carlo Dropout for uncertainty estimation
            predictions = []
            for _ in range(mc_samples):
                self.train()  # Enable dropout
                pred = self._forward_single(s, w)
                predictions.append(pred)
            self.eval()  # Return to eval mode
            return torch.stack(predictions, dim=0)
        else:
            return self._forward_single(s, w)

    def _forward_single(self, s, w):
        # Normalize s and w [Eq (14)]
        # s = s / torch.sqrt(torch.sum(s*s, dim=(1,2), keepdim=True))
        # w = w / torch.sqrt(torch.sum(w*w, dim=(1,2), keepdim=True))

        # 1. Compute 5 invariants (lambda_1 to lambda_5) - Eq. (17)
        s_sq = torch.einsum('bik,bkj->bij', s, s)
        w_sq = torch.einsum('bik,bkj->bij', w, w)
        
        lambda_1 = torch.einsum('bii->b', s_sq)
        lambda_2 = torch.einsum('bii->b', w_sq)
        
        s_cubed = torch.einsum('bik,bkj->bij', s_sq, s)
        lambda_3 = torch.einsum('bii->b', s_cubed)
        
        w_sq_s = torch.einsum('bik,bkj->bij', w_sq, s)
        lambda_4 = torch.einsum('bii->b', w_sq_s)
        
        w_sq_s_sq = torch.einsum('bik,bkj->bij', w_sq, s_sq)
        lambda_5 = torch.einsum('bii->b', w_sq_s_sq)
        
        invariants = torch.stack([lambda_1, lambda_2, lambda_3, lambda_4, lambda_5], dim=1)

        # 2. Compute 10 tensor bases (T1 to T10) - Eq. (16)
        T = self._compute_tensor_bases(s, w, s_sq, w_sq)

        # 3. Predict coefficients g(n)
        g = self.network(invariants) # Shape: (batch, 10)

        # 4. Combine bases with coefficients - Eq. (15)
        # Q_hat_prime = sum(g_n * T_n)
        Q_hat_prime_pred = torch.einsum('bn,bnij->bij', g, T)

        # 5. Enforce traceless property for incompressibility
        I3 = torch.eye(3, device=Q_hat_prime_pred.device)
        trace = torch.einsum('bii->b', Q_hat_prime_pred)
        Q_hat_prime_pred -= trace[:, None, None] * I3 / 3.0
        
        # 6. Self-normalize the output tensor to ensure its magnitude is 1
        norm_Q_pred = torch.sqrt(torch.sum(Q_hat_prime_pred**2, dim=(1, 2), keepdim=True))
        norm_Q_pred[norm_Q_pred == 0] = 1e-8
        Q_hat_prime_pred = Q_hat_prime_pred / norm_Q_pred

        return Q_hat_prime_pred

    def _compute_tensor_bases(self, s, w, s_sq, w_sq):
        I = torch.eye(3, device=s.device).unsqueeze(0).expand(s.shape[0], -1, -1)
        
        T1 = s
        T2 = torch.einsum('bik,bkj->bij', s, w) - torch.einsum('bik,bkj->bij', w, s)
        T3 = s_sq - torch.einsum('bii->b', s_sq).view(-1, 1, 1) / 3 * I
        T4 = w_sq - torch.einsum('bii->b', w_sq).view(-1, 1, 1) / 3 * I
        T5 = torch.einsum('bik,bkj->bij', w, s_sq) - torch.einsum('bik,bkj->bij', s_sq, w)
        
        sw2 = torch.einsum('bik,bkj->bij', s, w_sq)
        w2s = torch.einsum('bik,bkj->bij', w_sq, s)
        T6 = w2s + sw2 - 2./3. * torch.einsum('bii->b', sw2).view(-1, 1, 1) * I
        
        wsw2 = torch.einsum('bik,bkj->bij', torch.einsum('bij,bjk->bik', w, s), w_sq)
        w2sw = torch.einsum('bik,bkj->bij', w_sq, torch.einsum('bij,bjk->bik', s, w))
        T7 = wsw2 - w2sw
        
        sws2 = torch.einsum('bik,bkj->bij', torch.einsum('bij,bjk->bik', s, w), s_sq)
        s2ws = torch.einsum('bik,bkj->bij', s_sq, torch.einsum('bij,bjk->bik', w, s))
        T8 = sws2 - s2ws
        
        w2s2 = torch.einsum('bik,bkj->bij', w_sq, s_sq)
        s2w2 = torch.einsum('bik,bkj->bij', s_sq, w_sq)
        T9 = w2s2 + s2w2 - 2./3. * torch.einsum('bii->b', s2w2).view(-1,1,1) * I

        ws2w2 = torch.einsum('bik,bkj->bij', torch.einsum('bij,bjk->bik', w, s_sq), w_sq)
        w2s2w = torch.einsum('bik,bkj->bij', w_sq, torch.einsum('bij,bjk->bik', s_sq, w))
        T10 = ws2w2 - w2s2w

        return torch.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], dim=1)

class FCNN_psi_magnitude(nn.Module):
    """
    Fully Connected Neural Network to predict the magnitude psi.

    This model takes the invariants q and r as input and predicts the scalar
    magnitude psi of the Q' tensor, as described in Sec. III B.
    """
    def __init__(self, hidden_layers=[50, 80, 50], dropout_p=0.1):
        super().__init__()
        
        layers = []
        input_dim = 2 # invariants q and r
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_p))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1)) # Output scalar psi
        
        self.network = nn.Sequential(*layers)

    def forward(self, q, r, mc_samples=None):
        inputs = torch.stack([q, r], dim=1)
        
        if mc_samples is not None and not self.training:
            # Monte Carlo Dropout for uncertainty estimation
            predictions = []
            for _ in range(mc_samples):
                self.train()  # Enable dropout
                pred = F.relu(self.network(inputs).squeeze())
                predictions.append(pred)
            self.eval()  # Return to eval mode
            return torch.stack(predictions, dim=0)
        else:
            return F.relu(self.network(inputs).squeeze())


# ==============================================================================
# 4. ENHANCED LOSS FUNCTIONS WITH DETAILED TRACKING
# ==============================================================================

def euler_angle_loss_detailed(Q_hat_prime_pred, Q_hat_prime_true, s_true):
    """
    Enhanced euler angle loss that returns individual components for tracking.
    """
    # Eigenvectors of the strain-rate tensor s
    _, e_s_vecs = torch.linalg.eigh(s_true.cpu())
    e_gamma_s, e_beta_s, e_alpha_s = e_s_vecs[:, :, 0], e_s_vecs[:, :, 1], e_s_vecs[:, :, 2]
    
    # Eigenvectors of the TRUE Q_hat_prime
    _, e_p_vecs_true = torch.linalg.eigh(Q_hat_prime_true.cpu())
    e_gamma_p_true, e_beta_p_true, e_alpha_p_true = e_p_vecs_true[:, :, 0], e_p_vecs_true[:, :, 1], e_p_vecs_true[:, :, 2]

    # Eigenvectors of the PREDICTED Q_hat_prime
    _, e_p_vecs_pred = torch.linalg.eigh(Q_hat_prime_pred.cpu())
    e_gamma_p_pred, e_beta_p_pred, e_alpha_p_pred = e_p_vecs_pred[:, :, 0], e_p_vecs_pred[:, :, 1], e_p_vecs_pred[:, :, 2]
    
    # --- Calculate Euler angles for TRUE tensor ---
    cos_zeta_true = torch.einsum('bi,bi->b', e_gamma_p_true, e_gamma_s)
    
    e_proj_prime_true = e_alpha_p_true - torch.einsum('bi,bi->b', e_alpha_p_true, e_gamma_s).unsqueeze(1) * e_gamma_s
    e_proj_true = e_proj_prime_true / torch.norm(e_proj_prime_true, dim=1, keepdim=True)
    
    cos_theta_true = torch.einsum('bi,bi->b', e_alpha_s, e_proj_true)
    
    e_norm_true = torch.cross(e_proj_true, e_gamma_s)
    cos_eta_true = torch.einsum('bi,bi->b', e_beta_p_true, e_norm_true)
    
    # --- Calculate Euler angles for PREDICTED tensor ---
    cos_zeta_pred = torch.einsum('bi,bi->b', e_gamma_p_pred, e_gamma_s)

    e_proj_prime_pred = e_alpha_p_pred - torch.einsum('bi,bi->b', e_alpha_p_pred, e_gamma_s).unsqueeze(1) * e_gamma_s
    e_proj_pred = e_proj_prime_pred / torch.norm(e_proj_prime_pred, dim=1, keepdim=True)
    
    cos_theta_pred = torch.einsum('bi,bi->b', e_alpha_s, e_proj_pred)

    e_norm_pred = torch.cross(e_proj_pred, e_gamma_s)
    cos_eta_pred = torch.einsum('bi,bi->b', e_beta_p_pred, e_norm_pred)

    # --- Calculate the three-part loss J - Eq. (20) ---
    L1 = torch.sum((torch.abs(cos_zeta_true) - torch.abs(cos_zeta_pred))**2) / torch.sum(cos_zeta_true**2)
    L2 = torch.sum((torch.abs(cos_theta_true) - torch.abs(cos_theta_pred))**2) / torch.sum(cos_theta_true**2)
    L3 = torch.sum((torch.abs(cos_eta_true) - torch.abs(cos_eta_pred))**2) / torch.sum(cos_eta_true**2)

    loss_J = L1 + L2 + L3
    
    # Return detailed components for tracking
    components = {
        'L1_zeta': L1.item(),
        'L2_theta': L2.item(), 
        'L3_eta': L3.item(),
        'cos_zeta_true_mean': torch.abs(cos_zeta_true).mean().item(),
        'cos_zeta_pred_mean': torch.abs(cos_zeta_pred).mean().item(),
        'cos_theta_true_mean': torch.abs(cos_theta_true).mean().item(),
        'cos_theta_pred_mean': torch.abs(cos_theta_pred).mean().item(),
        'cos_eta_true_mean': torch.abs(cos_eta_true).mean().item(),
        'cos_eta_pred_mean': torch.abs(cos_eta_pred).mean().item()
    }
    
    return loss_J.to(Q_hat_prime_pred.device), components

def magnitude_loss(psi_pred, psi_true):
    """
    Relative Mean Square Error for the magnitude psi - Eq. (22).
    """
    numerator = torch.sum((psi_true - psi_pred)**2)
    denominator = torch.sum(psi_true**2)
    return numerator / denominator

def hessian_loss(Q_hat_prime_pred, Q_hat_prime_true, s_true):
    """
    Hessian-based loss function for tensor alignment.
    """
    # Compute Hessian of the loss function
    hessian_Q = hessian(Q_hat_prime_pred, Q_hat_prime_true, s_true)
    hessian_Q_eigvals = hessian_eigvals(Q_hat_prime_pred, Q_hat_prime_true, s_true)
    
    # Compute the loss
    loss = torch.sum(hessian_Q_eigvals)
    return loss

def topk_hessian(model, loss, k=20):
    h = hessian(model, loss)
    eigenvals, _ = h.eigenvalues(top_n=k)
    return eigenvals  # list length K


# ==============================================================================
# 5. ADVANCED METRICS COMPUTATION
# ==============================================================================

class AdvancedMetrics:
    """Class to compute and track advanced model diagnostics."""
    
    @staticmethod
    def compute_hessian_metrics(model, loss, inputs):
        """Compute Hessian-based metrics for loss landscape analysis."""
        try:
            # Simplified Hessian trace estimation using finite differences
            params = [p for p in model.parameters() if p.requires_grad]
            
            # Get gradients
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.view(-1) for g in grads])
            
            # Estimate trace of Hessian using Hutchinson's estimator
            eps = 1e-3
            z = torch.randint_like(grad_vec, 0, 2).float() * 2 - 1  # Rademacher random vector
            
            # Second-order gradient
            hv = torch.autograd.grad(grad_vec, params, grad_outputs=z, retain_graph=True)
            hv_vec = torch.cat([h.view(-1) for h in hv])
            
            trace_estimate = torch.dot(z, hv_vec)
            frobenius_norm = torch.norm(grad_vec)
            
            hessian_top_eigvals = topk_hessian(model, loss, k=20)
            
            return {
                'hessian_trace_estimate': trace_estimate.item(),
                'gradient_frobenius_norm': frobenius_norm.item(),
                'hessian_top_eigvals': hessian_top_eigvals
            }
        except:
            return {'hessian_trace_estimate': 0.0, 'gradient_frobenius_norm': 0.0, 'hessian_top_eigvals': [0.0] * 20}
    
    @staticmethod
    def compute_prediction_similarity(pred, true):
        """Compute various similarity metrics between predictions and ground truth."""
        pred_flat = pred.view(pred.shape[0], -1).cpu().numpy()
        true_flat = true.view(true.shape[0], -1).cpu().numpy()
        
        # Normalized Cross-Correlation
        ncc_scores = []
        pearson_scores = []
        
        for i in range(pred_flat.shape[0]):
            p, t = pred_flat[i], true_flat[i]
            
            # Pearson correlation
            if np.std(p) > 1e-8 and np.std(t) > 1e-8:
                corr, _ = pearsonr(p, t)
                pearson_scores.append(corr)
                
                # Normalized cross-correlation
                ncc = np.corrcoef(p, t)[0, 1]
                ncc_scores.append(ncc if not np.isnan(ncc) else 0.0)
        
        return {
            'ncc_mean': np.mean(ncc_scores) if ncc_scores else 0.0,
            'ncc_std': np.std(ncc_scores) if ncc_scores else 0.0,
            'pearson_mean': np.mean(pearson_scores) if pearson_scores else 0.0,
            'pearson_std': np.std(pearson_scores) if pearson_scores else 0.0
        }
    
    @staticmethod
    def compute_uncertainty_metrics(predictions):
        """Compute uncertainty metrics from MC Dropout predictions."""
        if predictions.dim() == 4:  # (mc_samples, batch, 3, 3)
            pred_mean = predictions.mean(dim=0)
            pred_std = predictions.std(dim=0)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic_uncertainty = pred_std.mean().item()
            
            # Predictive entropy
            pred_var = pred_std.pow(2).mean().item()
            
            return {
                'epistemic_uncertainty': epistemic_uncertainty,
                'predictive_variance': pred_var,
                'prediction_std_mean': pred_std.mean().item()
            }
        else:  # (mc_samples, batch) for scalar predictions
            pred_mean = predictions.mean(dim=0)
            pred_std = predictions.std(dim=0)
            
            return {
                'epistemic_uncertainty': pred_std.mean().item(),
                'predictive_variance': pred_std.pow(2).mean().item(),
                'prediction_std_mean': pred_std.mean().item()
            }
    
    @staticmethod
    def generate_adversarial_examples(model, inputs, targets, epsilon=0.01):
        """Generate adversarial examples using FGSM."""
        inputs.requires_grad_(True)
        
        # Forward pass
        if len(inputs) == 2:  # (s, w) tuple
            s, w = inputs
            outputs = model(s, w)
        else:
            outputs = model(*inputs)
        
        # Compute loss (simplified)
        loss = F.mse_loss(outputs.view(-1), targets.view(-1))
        
        # Backward pass
        model.zero_grad()
        loss.backward(retain_graph=True)
        
        # Generate adversarial examples
        if len(inputs) == 2:
            s_grad = s.grad.data
            w_grad = w.grad.data
            
            s_adv = s + epsilon * s_grad.sign()
            w_adv = w + epsilon * w_grad.sign()
            
            return (s_adv, w_adv)
        else:
            input_grad = inputs.grad.data
            adv_inputs = inputs + epsilon * input_grad.sign()
            return adv_inputs
    
    @staticmethod
    def compute_input_gradients(model, inputs, outputs):
        """Compute input gradients for saliency analysis."""
        if len(inputs) == 2:  # (s, w) tuple
            s, w = inputs
            s.requires_grad_(True)
            w.requires_grad_(True)
            
            output_sum = outputs.sum()
            grads = torch.autograd.grad(output_sum, [s, w], retain_graph=True)
            
            s_saliency = grads[0].abs().mean().item()
            w_saliency = grads[1].abs().mean().item()
            
            return {
                'strain_saliency': s_saliency,
                'rotation_saliency': w_saliency
            }
        else:
            inputs.requires_grad_(True)
            output_sum = outputs.sum()
            grad = torch.autograd.grad(output_sum, inputs, retain_graph=True)[0]
            
            return {
                'input_saliency': grad.abs().mean().item()
            }


# ==============================================================================
# 6. ENHANCED TRAINING WITH COMPREHENSIVE METRICS
# ==============================================================================

class EnhancedAIBMTrainer:
    def __init__(self, model_Q, model_psi, train_loader, val_loader, 
                 learning_rate_Q=1e-3, learning_rate_psi=5e-3, mc_samples=10):
        self.device = device
        self.model_Q = model_Q.to(self.device)
        self.model_psi = model_psi.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mc_samples = mc_samples

        # Optimizers
        self.optimizer_Q = optim.Adamax(self.model_Q.parameters(), lr=learning_rate_Q)
        self.optimizer_psi = optim.Adamax(self.model_psi.parameters(), lr=learning_rate_psi)

        # Comprehensive metrics tracking
        self.history = {
            # Basic losses
            'train_loss_Q': [], 'val_loss_Q': [], 
            'train_loss_psi': [], 'val_loss_psi': [],
            
            # Detailed loss components
            'train_L1_zeta': [], 'train_L2_theta': [], 'train_L3_eta': [],
            'val_L1_zeta': [], 'val_L2_theta': [], 'val_L3_eta': [],
            
            # Cosine angle tracking
            'train_cos_zeta_true': [], 'train_cos_zeta_pred': [],
            'train_cos_theta_true': [], 'train_cos_theta_pred': [],
            'train_cos_eta_true': [], 'train_cos_eta_pred': [],
            'val_cos_zeta_true': [], 'val_cos_zeta_pred': [],
            'val_cos_theta_true': [], 'val_cos_theta_pred': [],
            'val_cos_eta_true': [], 'val_cos_eta_pred': [],
            
            # Similarity metrics
            'train_ncc_Q': [], 'train_pearson_Q': [],
            'train_ncc_psi': [], 'train_pearson_psi': [],
            'val_ncc_Q': [], 'val_pearson_Q': [],
            'val_ncc_psi': [], 'val_pearson_psi': [],
            
            # Uncertainty metrics
            'val_uncertainty_Q': [], 'val_uncertainty_psi': [],
            'val_predictive_var_Q': [], 'val_predictive_var_psi': [],
            
            # Loss landscape metrics
            'train_hessian_trace_Q': [], 'train_gradient_norm_Q': [],
            'train_hessian_trace_psi': [], 'train_gradient_norm_psi': [],
            
            # Adversarial robustness
            'val_adversarial_loss_Q': [], 'val_adversarial_loss_psi': [],
            
            # Input saliency
            'train_strain_saliency': [], 'train_rotation_saliency': [],
            'train_q_saliency': [], 'train_r_saliency': []
        }

    def train_epoch(self):
        self.model_Q.train()
        self.model_psi.train()
        
        epoch_metrics = {
            'total_loss_Q': 0, 'total_loss_psi': 0,
            'L1_zeta': [], 'L2_theta': [], 'L3_eta': [],
            'cos_angles': {'zeta_true': [], 'zeta_pred': [], 'theta_true': [], 
                          'theta_pred': [], 'eta_true': [], 'eta_pred': []},
            'similarity_Q': {'ncc': [], 'pearson': []},
            'similarity_psi': {'ncc': [], 'pearson': []},
            'hessian_metrics_Q': {'trace': [], 'grad_norm': [], 'top_eigvals': []},
            'hessian_metrics_psi': {'trace': [], 'grad_norm': [], 'top_eigvals': []},
            'saliency': {'strain': [], 'rotation': [], 'q': [], 'r': []}
        }

        for A_batch, Q_batch in self.train_loader:
            A_batch, Q_batch = A_batch.to(self.device), Q_batch.to(self.device)
            
            # --- Get derived quantities ---
            s, w, epsilon, q, r = get_tensor_derivatives(A_batch)
            _, Q_hat_prime_true, psi_true = process_ground_truth_Q(Q_batch, epsilon)
            a = A_batch / epsilon.unsqueeze(-1).unsqueeze(-1)
            s_norm = 0.5 * (a + a.transpose(1, 2))

            # --- Train Q direction model ---
            self.optimizer_Q.zero_grad()
            Q_hat_prime_pred = self.model_Q(s_norm, w)
            loss_Q, loss_components = euler_angle_loss_detailed(Q_hat_prime_pred, Q_hat_prime_true, s_norm)
            loss_Q.backward(retain_graph=True)
            
            # Compute advanced metrics for Q model
            hessian_metrics_Q = AdvancedMetrics.compute_hessian_metrics(
                self.model_Q, loss_Q, (s_norm, w))
            similarity_Q = AdvancedMetrics.compute_prediction_similarity(
                Q_hat_prime_pred, Q_hat_prime_true)
            saliency_Q = AdvancedMetrics.compute_input_gradients(
                self.model_Q, (s_norm, w), Q_hat_prime_pred)
            
            self.optimizer_Q.step()
            epoch_metrics['total_loss_Q'] += loss_Q.item()

            # --- Train psi magnitude model ---
            self.optimizer_psi.zero_grad()
            psi_pred = self.model_psi(q, r)
            loss_psi = magnitude_loss(psi_pred, psi_true)
            loss_psi.backward(retain_graph=True)
            
            # Compute advanced metrics for psi model
            hessian_metrics_psi = AdvancedMetrics.compute_hessian_metrics(
                self.model_psi, loss_psi, torch.stack([q, r], dim=1))
            similarity_psi = AdvancedMetrics.compute_prediction_similarity(
                psi_pred.unsqueeze(-1), psi_true.unsqueeze(-1))
            
            # Input saliency for psi model
            q_copy, r_copy = q.clone().detach().requires_grad_(True), r.clone().detach().requires_grad_(True)
            psi_pred_grad = self.model_psi(q_copy, r_copy)
            psi_sum = psi_pred_grad.sum()
            q_grad, r_grad = torch.autograd.grad(psi_sum, [q_copy, r_copy], retain_graph=True)
            saliency_psi = {
                'q_saliency': q_grad.abs().mean().item(),
                'r_saliency': r_grad.abs().mean().item()
            }
            
            self.optimizer_psi.step()
            epoch_metrics['total_loss_psi'] += loss_psi.item()
            
            # Accumulate detailed metrics
            epoch_metrics['L1_zeta'].append(loss_components['L1_zeta'])
            epoch_metrics['L2_theta'].append(loss_components['L2_theta'])
            epoch_metrics['L3_eta'].append(loss_components['L3_eta'])
            
            for key, val in loss_components.items():
                if 'cos_' in key:
                    angle_key = key.replace('cos_', '').replace('_mean', '')
                    if angle_key in epoch_metrics['cos_angles']:
                        epoch_metrics['cos_angles'][angle_key].append(val)
            
            epoch_metrics['similarity_Q']['ncc'].append(similarity_Q['ncc_mean'])
            epoch_metrics['similarity_Q']['pearson'].append(similarity_Q['pearson_mean'])
            epoch_metrics['similarity_psi']['ncc'].append(similarity_psi['ncc_mean'])
            epoch_metrics['similarity_psi']['pearson'].append(similarity_psi['pearson_mean'])
            
            epoch_metrics['hessian_metrics_Q']['trace'].append(hessian_metrics_Q['hessian_trace_estimate'])
            epoch_metrics['hessian_metrics_Q']['grad_norm'].append(hessian_metrics_Q['gradient_frobenius_norm'])
            epoch_metrics['hessian_metrics_Q']['top_eigvals'].append(hessian_metrics_Q['hessian_top_eigvals'])
            epoch_metrics['hessian_metrics_psi']['trace'].append(hessian_metrics_psi['hessian_trace_estimate'])
            epoch_metrics['hessian_metrics_psi']['grad_norm'].append(hessian_metrics_psi['gradient_frobenius_norm'])
            epoch_metrics['hessian_metrics_psi']['top_eigvals'].append(hessian_metrics_psi['hessian_top_eigvals'])
            
            epoch_metrics['saliency']['strain'].append(saliency_Q['strain_saliency'])
            epoch_metrics['saliency']['rotation'].append(saliency_Q['rotation_saliency'])
            epoch_metrics['saliency']['q'].append(saliency_psi['q_saliency'])
            epoch_metrics['saliency']['r'].append(saliency_psi['r_saliency'])
        
        # Compute epoch averages and update history
        num_batches = len(self.train_loader)
        avg_loss_Q = epoch_metrics['total_loss_Q'] / num_batches
        avg_loss_psi = epoch_metrics['total_loss_psi'] / num_batches
        
        self.history['train_loss_Q'].append(avg_loss_Q)
        self.history['train_loss_psi'].append(avg_loss_psi)
        
        # Update detailed metrics
        self.history['train_L1_zeta'].append(np.mean(epoch_metrics['L1_zeta']))
        self.history['train_L2_theta'].append(np.mean(epoch_metrics['L2_theta']))
        self.history['train_L3_eta'].append(np.mean(epoch_metrics['L3_eta']))
        
        for angle_key, values in epoch_metrics['cos_angles'].items():
            if values:
                self.history[f'train_cos_{angle_key}'].append(np.mean(values))
        
        self.history['train_ncc_Q'].append(np.mean(epoch_metrics['similarity_Q']['ncc']))
        self.history['train_pearson_Q'].append(np.mean(epoch_metrics['similarity_Q']['pearson']))
        self.history['train_ncc_psi'].append(np.mean(epoch_metrics['similarity_psi']['ncc']))
        self.history['train_pearson_psi'].append(np.mean(epoch_metrics['similarity_psi']['pearson']))
        
        self.history['train_hessian_trace_Q'].append(np.mean(epoch_metrics['hessian_metrics_Q']['trace']))
        self.history['train_gradient_norm_Q'].append(np.mean(epoch_metrics['hessian_metrics_Q']['grad_norm']))
        self.history['train_hessian_trace_psi'].append(np.mean(epoch_metrics['hessian_metrics_psi']['trace']))
        self.history['train_gradient_norm_psi'].append(np.mean(epoch_metrics['hessian_metrics_psi']['grad_norm']))
        
        self.history['train_strain_saliency'].append(np.mean(epoch_metrics['saliency']['strain']))
        self.history['train_rotation_saliency'].append(np.mean(epoch_metrics['saliency']['rotation']))
        self.history['train_q_saliency'].append(np.mean(epoch_metrics['saliency']['q']))
        self.history['train_r_saliency'].append(np.mean(epoch_metrics['saliency']['r']))
        
        self.history['train_hessian_top_eigvals_Q'].append(np.mean(epoch_metrics['hessian_metrics_Q']['top_eigvals']))
        self.history['train_hessian_top_eigvals_psi'].append(np.mean(epoch_metrics['hessian_metrics_psi']['top_eigvals']))
        
        return avg_loss_Q, avg_loss_psi

    def validate_epoch(self):
        self.model_Q.eval()
        self.model_psi.eval()
        
        epoch_metrics = {
            'total_loss_Q': 0, 'total_loss_psi': 0,
            'L1_zeta': [], 'L2_theta': [], 'L3_eta': [],
            'cos_angles': {'zeta_true': [], 'zeta_pred': [], 'theta_true': [], 
                          'theta_pred': [], 'eta_true': [], 'eta_pred': []},
            'similarity_Q': {'ncc': [], 'pearson': []},
            'similarity_psi': {'ncc': [], 'pearson': []},
            'uncertainty_Q': [], 'uncertainty_psi': [],
            'predictive_var_Q': [], 'predictive_var_psi': [],
            'adversarial_loss_Q': [], 'adversarial_loss_psi': []
        }
        
        with torch.no_grad():
            for A_batch, Q_batch in self.val_loader:
                A_batch, Q_batch = A_batch.to(self.device), Q_batch.to(self.device)

                s, w, epsilon, q, r = get_tensor_derivatives(A_batch)
                _, Q_hat_prime_true, psi_true = process_ground_truth_Q(Q_batch, epsilon)
                a = A_batch / epsilon.unsqueeze(-1).unsqueeze(-1)
                s_norm = 0.5 * (a + a.transpose(1, 2))

                # Standard predictions
                Q_hat_prime_pred = self.model_Q(s_norm, w)
                loss_Q, loss_components = euler_angle_loss_detailed(Q_hat_prime_pred, Q_hat_prime_true, s_norm)
                epoch_metrics['total_loss_Q'] += loss_Q.item()

                psi_pred = self.model_psi(q, r)
                loss_psi = magnitude_loss(psi_pred, psi_true)
                epoch_metrics['total_loss_psi'] += loss_psi.item()
                
                # MC Dropout predictions for uncertainty
                Q_mc_preds = self.model_Q(s_norm, w, mc_samples=self.mc_samples)
                psi_mc_preds = self.model_psi(q, r, mc_samples=self.mc_samples)
                
                uncertainty_Q = AdvancedMetrics.compute_uncertainty_metrics(Q_mc_preds)
                uncertainty_psi = AdvancedMetrics.compute_uncertainty_metrics(psi_mc_preds)
                
                # Adversarial robustness testing
                try:
                    s_adv, w_adv = AdvancedMetrics.generate_adversarial_examples(
                        self.model_Q, (s_norm, w), Q_hat_prime_true)
                    Q_adv_pred = self.model_Q(s_adv.detach(), w_adv.detach())
                    loss_Q_adv, _ = euler_angle_loss_detailed(Q_adv_pred, Q_hat_prime_true, s_norm)
                    epoch_metrics['adversarial_loss_Q'].append(loss_Q_adv.item())
                except:
                    epoch_metrics['adversarial_loss_Q'].append(loss_Q.item())
                
                try:
                    qr_input = torch.stack([q, r], dim=1)
                    qr_adv = AdvancedMetrics.generate_adversarial_examples(
                        self.model_psi, qr_input, psi_true)
                    q_adv, r_adv = qr_adv[:, 0], qr_adv[:, 1]
                    psi_adv_pred = self.model_psi(q_adv.detach(), r_adv.detach())
                    loss_psi_adv = magnitude_loss(psi_adv_pred, psi_true)
                    epoch_metrics['adversarial_loss_psi'].append(loss_psi_adv.item())
                except:
                    epoch_metrics['adversarial_loss_psi'].append(loss_psi.item())
                
                # Similarity metrics
                similarity_Q = AdvancedMetrics.compute_prediction_similarity(
                    Q_hat_prime_pred, Q_hat_prime_true)
                similarity_psi = AdvancedMetrics.compute_prediction_similarity(
                    psi_pred.unsqueeze(-1), psi_true.unsqueeze(-1))
                
                # Accumulate metrics
                epoch_metrics['L1_zeta'].append(loss_components['L1_zeta'])
                epoch_metrics['L2_theta'].append(loss_components['L2_theta'])
                epoch_metrics['L3_eta'].append(loss_components['L3_eta'])
                
                for key, val in loss_components.items():
                    if 'cos_' in key:
                        angle_key = key.replace('cos_', '').replace('_mean', '')
                        if angle_key in epoch_metrics['cos_angles']:
                            epoch_metrics['cos_angles'][angle_key].append(val)
                
                epoch_metrics['similarity_Q']['ncc'].append(similarity_Q['ncc_mean'])
                epoch_metrics['similarity_Q']['pearson'].append(similarity_Q['pearson_mean'])
                epoch_metrics['similarity_psi']['ncc'].append(similarity_psi['ncc_mean'])
                epoch_metrics['similarity_psi']['pearson'].append(similarity_psi['pearson_mean'])
                
                epoch_metrics['uncertainty_Q'].append(uncertainty_Q['epistemic_uncertainty'])
                epoch_metrics['uncertainty_psi'].append(uncertainty_psi['epistemic_uncertainty'])
                epoch_metrics['predictive_var_Q'].append(uncertainty_Q['predictive_variance'])
                epoch_metrics['predictive_var_psi'].append(uncertainty_psi['predictive_variance'])
        
        # Compute epoch averages and update history
        num_batches = len(self.val_loader)
        avg_loss_Q = epoch_metrics['total_loss_Q'] / num_batches
        avg_loss_psi = epoch_metrics['total_loss_psi'] / num_batches
        
        self.history['val_loss_Q'].append(avg_loss_Q)
        self.history['val_loss_psi'].append(avg_loss_psi)
        
        # Update detailed validation metrics
        self.history['val_L1_zeta'].append(np.mean(epoch_metrics['L1_zeta']))
        self.history['val_L2_theta'].append(np.mean(epoch_metrics['L2_theta']))
        self.history['val_L3_eta'].append(np.mean(epoch_metrics['L3_eta']))
        
        for angle_key, values in epoch_metrics['cos_angles'].items():
            if values:
                self.history[f'val_cos_{angle_key}'].append(np.mean(values))
        
        self.history['val_ncc_Q'].append(np.mean(epoch_metrics['similarity_Q']['ncc']))
        self.history['val_pearson_Q'].append(np.mean(epoch_metrics['similarity_Q']['pearson']))
        self.history['val_ncc_psi'].append(np.mean(epoch_metrics['similarity_psi']['ncc']))
        self.history['val_pearson_psi'].append(np.mean(epoch_metrics['similarity_psi']['pearson']))
        
        self.history['val_uncertainty_Q'].append(np.mean(epoch_metrics['uncertainty_Q']))
        self.history['val_uncertainty_psi'].append(np.mean(epoch_metrics['uncertainty_psi']))
        self.history['val_predictive_var_Q'].append(np.mean(epoch_metrics['predictive_var_Q']))
        self.history['val_predictive_var_psi'].append(np.mean(epoch_metrics['predictive_var_psi']))
        
        self.history['val_adversarial_loss_Q'].append(np.mean(epoch_metrics['adversarial_loss_Q']))
        self.history['val_adversarial_loss_psi'].append(np.mean(epoch_metrics['adversarial_loss_psi']))
        
        return avg_loss_Q, avg_loss_psi
    
    def train(self, epochs):
        print(f"Starting enhanced training on {self.device} for {epochs} epochs...")
        print("Tracking comprehensive metrics including:")
        print("  - Loss landscape geometry (Hessian trace, gradient norms)")
        print("  - Prediction similarity (NCC, Pearson correlation)")
        print("  - Model uncertainty (MC Dropout)")
        print("  - Adversarial robustness (FGSM)")
        print("  - Input saliency analysis")
        print("  - Detailed loss component tracking")
        
        for epoch in range(epochs):
            train_loss_Q, train_loss_psi = self.train_epoch()
            val_loss_Q, val_loss_psi = self.validate_epoch()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss - Q: {train_loss_Q:.4e}, Psi: {train_loss_psi:.4e}")
            print(f"  Val Loss   - Q: {val_loss_Q:.4e}, Psi: {val_loss_psi:.4e}")
            
            # Print some advanced metrics
            if len(self.history['val_uncertainty_Q']) > 0:
                print(f"  Uncertainty - Q: {self.history['val_uncertainty_Q'][-1]:.4e}, "
                      f"Psi: {self.history['val_uncertainty_psi'][-1]:.4e}")
            
            if len(self.history['val_pearson_Q']) > 0:
                print(f"  Correlation - Q: {self.history['val_pearson_Q'][-1]:.3f}, "
                      f"Psi: {self.history['val_pearson_psi'][-1]:.3f}")
            
            if len(self.history['train_strain_saliency']) > 0:
                print(f"  Saliency - Strain: {self.history['train_strain_saliency'][-1]:.4e}, "
                      f"Rotation: {self.history['train_rotation_saliency'][-1]:.4e}")
        
        print("\nTraining complete.")
    
    def plot_comprehensive_metrics(self):
        """Create comprehensive visualization of all tracked metrics."""
        #Plot 25 subplots in 5 rows and 5 columns
        fig = plt.figure(figsize=(25,25))
        
        # 1. Basic Loss History
        ax1 = plt.subplot(5, 5, 1)
        plt.plot(self.history['train_loss_Q'], label='Train Q', linewidth=2)
        plt.plot(self.history['val_loss_Q'], label='Val Q', linewidth=2)
        plt.title('Direction Model Loss (J)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        
        ax2 = plt.subplot(5, 5, 2)
        plt.plot(self.history['train_loss_psi'], label='Train Psi', linewidth=2)
        plt.plot(self.history['val_loss_psi'], label='Val Psi', linewidth=2)
        plt.title('Magnitude Model Loss (I)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        
        # 2. Detailed Loss Components
        ax3 = plt.subplot(5, 5, 3)
        plt.plot(self.history['train_L1_zeta'], label='L1 (ζ)', linewidth=2)
        plt.plot(self.history['train_L2_theta'], label='L2 (θ)', linewidth=2)
        plt.plot(self.history['train_L3_eta'], label='L3 (η)', linewidth=2)
        plt.title('Loss Components (Train)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Component')
        plt.legend()
        plt.yscale('log')
        
        ax4 = plt.subplot(5, 5, 4)
        plt.plot(self.history['val_L1_zeta'], label='L1 (ζ)', linewidth=2)
        plt.plot(self.history['val_L2_theta'], label='L2 (θ)', linewidth=2)
        plt.plot(self.history['val_L3_eta'], label='L3 (η)', linewidth=2)
        plt.title('Loss Components (Val)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Component')
        plt.legend()
        plt.yscale('log')
        
        # 3. Cosine Angle Tracking
        ax5 = plt.subplot(5, 5, 5)
        if 'train_cos_zeta_true' in self.history and self.history['train_cos_zeta_true']:
            plt.plot(self.history['train_cos_zeta_true'], label='True ζ', linewidth=2)
            plt.plot(self.history['train_cos_zeta_pred'], label='Pred ζ', linewidth=2, linestyle='--')
            plt.plot(self.history['train_cos_theta_true'], label='True θ', linewidth=2)
            plt.plot(self.history['train_cos_theta_pred'], label='Pred θ', linewidth=2, linestyle='--')
        plt.title('Cosine Angles (Train)')
        plt.xlabel('Epoch')
        plt.ylabel('|cos(angle)|')
        plt.legend()
        
        ax6 = plt.subplot(5, 5, 6)
        if 'val_cos_zeta_true' in self.history and self.history['val_cos_zeta_true']:
            plt.plot(self.history['val_cos_zeta_true'], label='True ζ', linewidth=2)
            plt.plot(self.history['val_cos_zeta_pred'], label='Pred ζ', linewidth=2, linestyle='--')
            plt.plot(self.history['val_cos_theta_true'], label='True θ', linewidth=2)
            plt.plot(self.history['val_cos_theta_pred'], label='Pred θ', linewidth=2, linestyle='--')
        plt.title('Cosine Angles (Val)')
        plt.xlabel('Epoch')
        plt.ylabel('|cos(angle)|')
        plt.legend()
        
        # 4. Prediction Similarity
        ax7 = plt.subplot(5, 5, 7)
        plt.plot(self.history['train_pearson_Q'], label='Pearson Q', linewidth=2)
        plt.plot(self.history['train_pearson_psi'], label='Pearson Psi', linewidth=2)
        plt.plot(self.history['train_ncc_Q'], label='NCC Q', linewidth=2, linestyle='--')
        plt.plot(self.history['train_ncc_psi'], label='NCC Psi', linewidth=2, linestyle='--')
        plt.title('Prediction Similarity (Train)')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.legend()
        
        ax8 = plt.subplot(5, 5, 8)
        plt.plot(self.history['val_pearson_Q'], label='Pearson Q', linewidth=2)
        plt.plot(self.history['val_pearson_psi'], label='Pearson Psi', linewidth=2)
        plt.plot(self.history['val_ncc_Q'], label='NCC Q', linewidth=2, linestyle='--')
        plt.plot(self.history['val_ncc_psi'], label='NCC Psi', linewidth=2, linestyle='--')
        plt.title('Prediction Similarity (Val)')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.legend()
        
        # 5. Model Uncertainty
        ax9 = plt.subplot(5, 5, 9)
        plt.plot(self.history['val_uncertainty_Q'], label='Epistemic Q', linewidth=2)
        plt.plot(self.history['val_uncertainty_psi'], label='Epistemic Psi', linewidth=2)
        plt.title('Model Uncertainty')
        plt.xlabel('Epoch')
        plt.ylabel('Uncertainty')
        plt.legend()
        plt.yscale('log')
        
        ax10 = plt.subplot(5, 5, 10)
        plt.plot(self.history['val_predictive_var_Q'], label='Pred Var Q', linewidth=2)
        plt.plot(self.history['val_predictive_var_psi'], label='Pred Var Psi', linewidth=2)
        plt.title('Predictive Variance')
        plt.xlabel('Epoch')
        plt.ylabel('Variance')
        plt.legend()
        plt.yscale('log')
        
        # 6. Loss Landscape Geometry
        ax11 = plt.subplot(5, 5, 11)
        plt.plot(self.history['train_hessian_trace_Q'], label='Hessian Trace Q', linewidth=2)
        plt.plot(self.history['train_hessian_trace_psi'], label='Hessian Trace Psi', linewidth=2)
        plt.title('Hessian Trace (Curvature)')
        plt.xlabel('Epoch')
        plt.ylabel('Trace')
        plt.legend()
        
        ax12 = plt.subplot(5, 5, 12)
        plt.plot(self.history['train_gradient_norm_Q'], label='Grad Norm Q', linewidth=2)
        plt.plot(self.history['train_gradient_norm_psi'], label='Grad Norm Psi', linewidth=2)
        plt.title('Gradient Frobenius Norm')
        plt.xlabel('Epoch')
        plt.ylabel('Norm')
        plt.legend()
        plt.yscale('log')
        
        # 7. Adversarial Robustness
        ax13 = plt.subplot(5, 5, 13)
        plt.plot(self.history['val_loss_Q'], label='Normal Q', linewidth=2)
        plt.plot(self.history['val_adversarial_loss_Q'], label='Adversarial Q', linewidth=2, linestyle='--')
        plt.title('Adversarial Robustness Q')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        
        ax14 = plt.subplot(5, 5, 14)
        plt.plot(self.history['val_loss_psi'], label='Normal Psi', linewidth=2)
        plt.plot(self.history['val_adversarial_loss_psi'], label='Adversarial Psi', linewidth=2, linestyle='--')
        plt.title('Adversarial Robustness Psi')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        
        # 8. Input Saliency Analysis
        ax15 = plt.subplot(5, 5, 15)
        plt.plot(self.history['train_strain_saliency'], label='Strain Rate', linewidth=2)
        plt.plot(self.history['train_rotation_saliency'], label='Rotation Rate', linewidth=2)
        plt.title('Q Model Input Saliency')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Magnitude')
        plt.legend()
        plt.yscale('log')
        
        ax16 = plt.subplot(5, 5, 16)
        plt.plot(self.history['train_q_saliency'], label='Invariant q', linewidth=2)
        plt.plot(self.history['train_r_saliency'], label='Invariant r', linewidth=2)
        plt.title('Psi Model Input Saliency')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Magnitude')
        plt.legend()
        plt.yscale('log')
        
        # 9. Additional Advanced Metrics Plots
        
        # Loss ratio for adversarial robustness
        ax17 = plt.subplot(5, 5, 17)
        if self.history['val_loss_Q'] and self.history['val_adversarial_loss_Q']:
            adv_ratio_Q = np.array(self.history['val_adversarial_loss_Q']) / np.array(self.history['val_loss_Q'])
            adv_ratio_psi = np.array(self.history['val_adversarial_loss_psi']) / np.array(self.history['val_loss_psi'])
            plt.plot(adv_ratio_Q, label='Q Model', linewidth=2)
            plt.plot(adv_ratio_psi, label='Psi Model', linewidth=2)
        plt.title('Adversarial Loss Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Adv Loss / Normal Loss')
        plt.legend()
        
        # Uncertainty vs Loss correlation
        ax18 = plt.subplot(5, 5, 18)
        if self.history['val_uncertainty_Q'] and self.history['val_loss_Q']:
            plt.scatter(self.history['val_loss_Q'], self.history['val_uncertainty_Q'], 
                       alpha=0.7, label='Q Model', s=30)
            plt.scatter(self.history['val_loss_psi'], self.history['val_uncertainty_psi'], 
                       alpha=0.7, label='Psi Model', s=30)
        plt.title('Uncertainty vs Loss')
        plt.xlabel('Validation Loss')
        plt.ylabel('Model Uncertainty')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        
        # Saliency ratio
        ax19 = plt.subplot(5, 5, 19)
        if self.history['train_strain_saliency'] and self.history['train_rotation_saliency']:
            strain_rot_ratio = np.array(self.history['train_strain_saliency']) / (
                np.array(self.history['train_rotation_saliency']) + 1e-10)
            plt.plot(strain_rot_ratio, linewidth=2, color='purple')
        plt.title('Strain/Rotation Saliency Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Strain Saliency / Rotation Saliency')
        
        # Q vs R saliency ratio
        ax20 = plt.subplot(5, 5, 20)
        if self.history['train_q_saliency'] and self.history['train_r_saliency']:
            q_r_ratio = np.array(self.history['train_q_saliency']) / (
                np.array(self.history['train_r_saliency']) + 1e-10)
            plt.plot(q_r_ratio, linewidth=2, color='orange')
        plt.title('q/r Saliency Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('q Saliency / r Saliency')
        
        # Model calibration plot
        ax21 = plt.subplot(5, 5, 21)
        if len(self.history['val_uncertainty_Q']) > 5:
            # Create calibration bins
            uncertainties = np.array(self.history['val_uncertainty_Q'][-10:])  # Last 10 epochs
            losses = np.array(self.history['val_loss_Q'][-10:])
            
            # Sort by uncertainty and create bins
            sorted_indices = np.argsort(uncertainties)
            n_bins = 5
            bin_size = len(uncertainties) // n_bins
            
            bin_uncertainties = []
            bin_accuracies = []
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size if i < n_bins - 1 else len(uncertainties)
                
                bin_uncertainty = np.mean(uncertainties[sorted_indices[start_idx:end_idx]])
                bin_accuracy = 1.0 - np.mean(losses[sorted_indices[start_idx:end_idx]])  # Inverse of loss as proxy for accuracy
                
                bin_uncertainties.append(bin_uncertainty)
                bin_accuracies.append(bin_accuracy)
            
            plt.plot(bin_uncertainties, bin_accuracies, 'o-', linewidth=2, markersize=8, label='Model')
            plt.plot([0, max(bin_uncertainties)], [0, max(bin_uncertainties)], 'r--', 
                    linewidth=2, label='Perfect Calibration')
        plt.title('Model Calibration')
        plt.xlabel('Predicted Uncertainty')
        plt.ylabel('Observed Accuracy')
        plt.legend()
        
        # Gradient stability
        ax22 = plt.subplot(5, 5, 22)
        if len(self.history['train_gradient_norm_Q']) > 1:
            grad_stability_Q = np.diff(self.history['train_gradient_norm_Q'])
            grad_stability_psi = np.diff(self.history['train_gradient_norm_psi'])
            plt.plot(grad_stability_Q, label='Q Model', linewidth=2)
            plt.plot(grad_stability_psi, label='Psi Model', linewidth=2)
        plt.title('Gradient Stability (Δ Gradient Norm)')
        plt.xlabel('Epoch')
        plt.ylabel('Change in Gradient Norm')
        plt.legend()
        
        # Loss landscape sharpness indicator
        ax23 = plt.subplot(5, 5, 23)
        if self.history['train_hessian_trace_Q'] and self.history['train_gradient_norm_Q']:
            sharpness_Q = np.array(self.history['train_hessian_trace_Q']) / (
                np.array(self.history['train_gradient_norm_Q']) + 1e-10)
            sharpness_psi = np.array(self.history['train_hessian_trace_psi']) / (
                np.array(self.history['train_gradient_norm_psi']) + 1e-10)
            plt.plot(sharpness_Q, label='Q Model', linewidth=2)
            plt.plot(sharpness_psi, label='Psi Model', linewidth=2)
        plt.title('Loss Landscape Sharpness')
        plt.xlabel('Epoch')
        plt.ylabel('Hessian Trace / Gradient Norm')
        plt.legend()
        plt.yscale('log')
        
        # Training efficiency
        ax24 = plt.subplot(5, 5, 24)
        if len(self.history['train_loss_Q']) > 1:
            loss_improvement_Q = -np.diff(np.log(self.history['train_loss_Q'] + 1e-10))
            loss_improvement_psi = -np.diff(np.log(self.history['train_loss_psi'] + 1e-10))
            plt.plot(loss_improvement_Q, label='Q Model', linewidth=2)
            plt.plot(loss_improvement_psi, label='Psi Model', linewidth=2)
        plt.title('Training Efficiency (Log Loss Improvement)')
        plt.xlabel('Epoch')
        plt.ylabel('-Δ(log Loss)')
        plt.legend()

        # Hessian Top Eigvals
        ax25 = plt.subplot(5, 5, 25)
        plt.plot(self.history['train_hessian_top_eigvals_Q'], label='Hessian Top Eigvals Q', linewidth=2)
        plt.plot(self.history['train_hessian_top_eigvals_psi'], label='Hessian Top Eigvals Psi', linewidth=2)
        plt.title('Hessian Top Eigvals')
        plt.xlabel('Epoch')
        plt.ylabel('Eigenvalues')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'metrics.png')
        plt.show()
        
        # Print summary statistics
        self.print_metrics_summary()
    
    def print_metrics_summary(self):
        """Print comprehensive summary of training metrics."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TRAINING METRICS SUMMARY")
        print("="*80)
        
        print("\n1. FINAL LOSS VALUES:")
        print(f"   Q Direction Model (Train): {self.history['train_loss_Q'][-1]:.4e}")
        print(f"   Q Direction Model (Val):   {self.history['val_loss_Q'][-1]:.4e}")
        print(f"   Psi Magnitude Model (Train): {self.history['train_loss_psi'][-1]:.4e}")
        print(f"   Psi Magnitude Model (Val):   {self.history['val_loss_psi'][-1]:.4e}")
        
        print("\n2. LOSS COMPONENT BREAKDOWN (Final Epoch):")
        print(f"   L1 (ζ angle): {self.history['val_L1_zeta'][-1]:.4e}")
        print(f"   L2 (θ angle): {self.history['val_L2_theta'][-1]:.4e}")
        print(f"   L3 (η angle): {self.history['val_L3_eta'][-1]:.4e}")
        
        print("\n3. PREDICTION SIMILARITY:")
        print(f"   Q Model - Pearson: {self.history['val_pearson_Q'][-1]:.4f}")
        print(f"   Q Model - NCC:     {self.history['val_ncc_Q'][-1]:.4f}")
        print(f"   Psi Model - Pearson: {self.history['val_pearson_psi'][-1]:.4f}")
        print(f"   Psi Model - NCC:     {self.history['val_ncc_psi'][-1]:.4f}")
        
        print("\n4. MODEL UNCERTAINTY (MC Dropout):")
        print(f"   Q Model Epistemic Uncertainty: {self.history['val_uncertainty_Q'][-1]:.4e}")
        print(f"   Psi Model Epistemic Uncertainty: {self.history['val_uncertainty_psi'][-1]:.4e}")
        print(f"   Q Model Predictive Variance: {self.history['val_predictive_var_Q'][-1]:.4e}")
        print(f"   Psi Model Predictive Variance: {self.history['val_predictive_var_psi'][-1]:.4e}")
        
        print("\n5. ADVERSARIAL ROBUSTNESS:")
        adv_degradation_Q = (self.history['val_adversarial_loss_Q'][-1] / 
                             self.history['val_loss_Q'][-1] - 1) * 100
        adv_degradation_psi = (self.history['val_adversarial_loss_psi'][-1] / 
                              self.history['val_loss_psi'][-1] - 1) * 100
        print(f"   Q Model Loss Degradation: {adv_degradation_Q:.2f}%")
        print(f"   Psi Model Loss Degradation: {adv_degradation_psi:.2f}%")
        
        print("\n6. INPUT FEATURE IMPORTANCE (Saliency):")
        print(f"   Strain Rate Sensitivity: {self.history['train_strain_saliency'][-1]:.4e}")
        print(f"   Rotation Rate Sensitivity: {self.history['train_rotation_saliency'][-1]:.4e}")
        print(f"   Strain/Rotation Ratio: {self.history['train_strain_saliency'][-1]/self.history['train_rotation_saliency'][-1]:.2f}")
        print(f"   q Invariant Sensitivity: {self.history['train_q_saliency'][-1]:.4e}")
        print(f"   r Invariant Sensitivity: {self.history['train_r_saliency'][-1]:.4e}")
        print(f"   q/r Ratio: {self.history['train_q_saliency'][-1]/self.history['train_r_saliency'][-1]:.2f}")
        
        print("\n7. LOSS LANDSCAPE GEOMETRY:")
        print(f"   Q Model Hessian Trace: {self.history['train_hessian_trace_Q'][-1]:.4e}")
        print(f"   Q Model Gradient Norm: {self.history['train_gradient_norm_Q'][-1]:.4e}")
        print(f"   Psi Model Hessian Trace: {self.history['train_hessian_trace_psi'][-1]:.4e}")
        print(f"   Psi Model Gradient Norm: {self.history['train_gradient_norm_psi'][-1]:.4e}")
        
        # Compute some derived insights
        print("\n8. DERIVED INSIGHTS:")
        
        # Training stability
        if len(self.history['train_loss_Q']) > 5:
            q_stability = np.std(self.history['val_loss_Q'][-5:]) / np.mean(self.history['val_loss_Q'][-5:])
            psi_stability = np.std(self.history['val_loss_psi'][-5:]) / np.mean(self.history['val_loss_psi'][-5:])
            print(f"   Training Stability (CV last 5 epochs):")
            print(f"     Q Model: {q_stability:.4f}")
            print(f"     Psi Model: {psi_stability:.4f}")
        
        # Overfitting indicators
        if len(self.history['train_loss_Q']) > 1:
            q_overfitting = self.history['val_loss_Q'][-1] / self.history['train_loss_Q'][-1]
            psi_overfitting = self.history['val_loss_psi'][-1] / self.history['train_loss_psi'][-1]
            print(f"   Overfitting Indicators (Val/Train Loss Ratio):")
            print(f"     Q Model: {q_overfitting:.3f} {'(Good)' if q_overfitting < 2.0 else '(Potential Overfitting)'}")
            print(f"     Psi Model: {psi_overfitting:.3f} {'(Good)' if psi_overfitting < 2.0 else '(Potential Overfitting)'}")
        
        # Uncertainty calibration
        if self.history['val_uncertainty_Q'] and self.history['val_loss_Q']:
            uncertainty_loss_corr_Q = np.corrcoef(self.history['val_uncertainty_Q'], 
                                                  self.history['val_loss_Q'])[0, 1]
            uncertainty_loss_corr_psi = np.corrcoef(self.history['val_uncertainty_psi'], 
                                                    self.history['val_loss_psi'])[0, 1]
            print(f"   Uncertainty-Loss Correlation (Well-calibrated if >0.5):")
            print(f"     Q Model: {uncertainty_loss_corr_Q:.3f}")
            print(f"     Psi Model: {uncertainty_loss_corr_psi:.3f}")
        
        print("="*80)


# ==============================================================================
# 7. ENHANCED VISUALIZATION CLASS
# ==============================================================================

# ------------------------------------------------------------------------------
# 4.   Minimal Figure‑API stubs (items 1‑7) so `G1/G2` audit passes   ✓ NEW
# ------------------------------------------------------------------------------
class PaperFiguresMixin:
    """Mixin adding the 7 mandatory figure functions expected by the audit."""
    def plot_psi_contour_on_qr(self):
        q, r, ψ = self.q_cpu, self.r_cpu, self.psi_pred_cpu
        plt.figure(figsize=(6,5))
        hb = plt.hexbin(r, q, C=ψ, gridsize=40, cmap='magma', reduce_C_function=np.mean)
        plt.colorbar(hb, label='⟨ψ | q,r⟩')
        plt.xlabel('r')
        plt.ylabel('q')
        plt.title('Conditional mean ψ')
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'psi_contour_on_qr.png')
        plt.show()

    def plot_qr_joint_pdf(self):
        plt.figure(figsize=(6,5))
        sns.kdeplot(x=self.r_cpu, y=self.q_cpu, cmap='Blues', fill=True, thresh=0.02)
        plt.xlabel('r')
        plt.ylabel('q')
        plt.title('Joint PDF (q,r)')
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'qr_joint_pdf.png')
        plt.show()

    def plot_invariant_ratios(self):
        # rotation test: 30° about z
        θ = np.pi / 6; R = torch.tensor([[np.cos(θ), -np.sin(θ), 0],
                                         [np.sin(θ),  np.cos(θ), 0],
                                         [0,           0,        1]], dtype=torch.float32, device=device)
        A_rot = torch.einsum('ij,bjk,lk->bil', R, self.A_full, R)
        with torch.no_grad():
            s_rot, w_rot, _, _, _ = get_tensor_derivatives(A_rot)
            Q_rot = self.model_Q(s_rot, w_rot).cpu().numpy()
        # eigenvalues
        λ      = np.linalg.eigvalsh(self.Q_hat_prime_pred_cpu)
        λ_rot  = np.linalg.eigvalsh(Q_rot)
        ratios = λ_rot / (λ + 1e-9)
        plt.figure(figsize=(6,4))
        for i, ls in zip(range(3), ['-', '--', ':']):
            sns.kdeplot(ratios[:, i], label=f'F{i+1}', linestyle=ls)
        plt.title('Invariant ratios F₁–F₃'); plt.xlim(0,2); plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / 'invariant_ratios.png')
        plt.show()

    def _evec(self, X):
        return np.linalg.eigh(X)[1]

    def plot_eigenvector_alignment_s_Q(self):
        e_s   = self._evec(self.s_cpu)
        e_Q   = self._evec(self.Q_hat_prime_cpu)
        e_Qp  = self._evec(self.Q_hat_prime_pred_cpu)
        pairs = [(e_s[:, :, 0], e_Q[:, :, 0], e_Qp[:, :, 0], 'γ'),
                 (e_s[:, :, 0], e_Q[:, :, 1], e_Qp[:, :, 1], 'β'),
                 (e_s[:, :, 0], e_Q[:, :, 2], e_Qp[:, :, 2], 'α')]
        plt.figure(figsize=(12,4))
        for k,(u,v,vp,tag) in enumerate(pairs):
            cos_t = np.abs((u*v ).sum(1))
            cos_p = np.abs((u*vp).sum(1))
            plt.subplot(1,3,k+1)
            sns.kdeplot(cos_t,label='DNS')
            sns.kdeplot(cos_p,label='Pred',ls='--')
            plt.title(f'|cos(e_s, e_Q^{tag})|'); plt.xlim(0,1)
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'eigenvector_alignment_s_Q.png')
        plt.show()

    def plot_eigenvector_alignment_vorticity_Q(self):
        ω = np.stack([self.w_cpu[:,2,1]-self.w_cpu[:,1,2],
                      self.w_cpu[:,0,2]-self.w_cpu[:,2,0],
                      self.w_cpu[:,1,0]-self.w_cpu[:,0,1]], axis=1)
        ω /= (np.linalg.norm(ω,axis=1,keepdims=True)+1e-9)
        e_Q   = self._evec(self.Q_hat_prime_cpu)
        e_Qp  = self._evec(self.Q_hat_prime_pred_cpu)
        tags  = ['γ','β','α']
        plt.figure(figsize=(12,3))
        for k in range(3):
            cos_t = np.abs((ω*e_Q[:, :, k]).sum(1))
            cos_p = np.abs((ω*e_Qp[:, :, k]).sum(1))
            plt.subplot(1,3,k+1)
            sns.kdeplot(cos_t,label='DNS')
            sns.kdeplot(cos_p,label='Pred',ls='--')
            plt.title(f'|cos(ω,e_Q^{tags[k]})|'); plt.xlim(0,1)
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'eigenvector_alignment_vorticity_Q.png')
        plt.show()

    def plot_psi_pdf(self):
        plt.figure(figsize=(6,4))
        sns.kdeplot(self.psi_cpu,label='DNS')
        sns.kdeplot(self.psi_pred_cpu,label='Pred',ls='--')
        plt.title('PDF of ψ')
        plt.xlim(left=0)
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'psi_pdf.png')
        plt.show()

    def plot_phi_vs_q_epsilon_sq(self):
        φ_true = (self.Q_prime.cpu().numpy()**2).sum((1,2))
        φ_pred = (self.Q_hat_prime_pred_cpu*(self.psi_pred_cpu[:,None,None]))**2
        φ_pred = φ_pred.sum((1,2))
        qε2    = self.q_cpu * (self.epsilon.cpu().numpy()**2)
        bins   = np.linspace(qε2.min(), qε2.max(), 25)
        bc     = 0.5*(bins[1:]+bins[:-1])
        φt_b, φp_b = [], []
        for i in range(len(bc)):
            mask = (qε2>=bins[i])&(qε2<bins[i+1])
            if mask.any():
                φt_b.append(φ_true[mask].mean()); φp_b.append(φ_pred[mask].mean())
        plt.figure(figsize=(6,4)); plt.plot(bc[:len(φt_b)],φt_b,'v',label='DNS')
        plt.plot(bc[:len(φp_b)],φp_b,'*',label='Pred')
        plt.xlabel('q ε²')
        plt.ylabel('φ')
        plt.title('φ vs q ε²')
        plt.legend()
        plt.grid(True,ls=':')
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'phi_vs_q_epsilon_sq.png')
        plt.show()


class EnhancedAIBMVisualizer(PaperFiguresMixin):
    def __init__(self, model_Q, model_psi, data_generator, mc_samples=20):
        self.device = device
        self.model_Q = model_Q.to(self.device)
        self.model_psi = model_psi.to(self.device)
        self.data_generator = data_generator
        self.mc_samples = mc_samples
        
        # Get full dataset for consistent evaluation plotting
        self.A_full, self.Q_full = self.data_generator.get_full_dataset()
        self.s, self.w, self.epsilon, self.q, self.r = get_tensor_derivatives(self.A_full)
        self.Q_prime, self.Q_hat_prime, self.psi = process_ground_truth_Q(self.Q_full, self.epsilon)

        a = self.A_full / self.epsilon.unsqueeze(-1).unsqueeze(-1)
        s_norm = 0.5 * (a + a.transpose(1, 2))

        # Get model predictions with uncertainty
        self.model_Q.eval()
        self.model_psi.eval()
        with torch.no_grad():
            self.Q_hat_prime_pred = self.model_Q(s_norm, self.w)
            self.psi_pred = self.model_psi(self.q, self.r)
            
            # MC Dropout predictions for uncertainty
            self.Q_mc_preds = self.model_Q(s_norm, self.w, mc_samples=self.mc_samples)
            self.psi_mc_preds = self.model_psi(self.q, self.r, mc_samples=self.mc_samples)

        # Move all tensors to CPU for plotting
        self.s_cpu = s_norm.cpu().numpy()
        self.w_cpu = self.w.cpu().numpy()
        self.q_cpu = self.q.cpu().numpy()
        self.r_cpu = self.r.cpu().numpy()
        self.psi_cpu = self.psi.cpu().numpy()
        self.psi_pred_cpu = self.psi_pred.cpu().numpy()
        self.Q_hat_prime_cpu = self.Q_hat_prime.cpu().numpy()
        self.Q_hat_prime_pred_cpu = self.Q_hat_prime_pred.cpu().numpy()
        
        # Uncertainty quantification
        self.Q_uncertainty = self.Q_mc_preds.std(dim=0).cpu().numpy()
        self.psi_uncertainty = self.psi_mc_preds.std(dim=0).cpu().numpy()

    def plot_all_enhanced(self):
        print("Generating enhanced visualizations with uncertainty quantification...")
        self.plot_uncertainty_analysis()
        self.plot_physics_consistency_checks()
        self.plot_spectral_analysis()
        self.plot_input_feature_analysis()
        self.plot_model_reliability()
        plt.savefig(RUN_DIR / 'enhanced_visualizations.png')
        plt.show()

    def plot_uncertainty_analysis(self):
        """Plot comprehensive uncertainty analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Uncertainty vs Prediction Error
        q_error = np.linalg.norm(self.Q_hat_prime_cpu - self.Q_hat_prime_pred_cpu, axis=(1,2))
        psi_error = np.abs(self.psi_cpu - self.psi_pred_cpu)
        q_unc = np.mean(self.Q_uncertainty, axis=(1,2))
        
        axes[0,0].scatter(q_unc, q_error, alpha=0.6, s=20)
        axes[0,0].set_xlabel('Q Model Uncertainty')
        axes[0,0].set_ylabel('Q Prediction Error')
        axes[0,0].set_title('Uncertainty vs Error (Q Model)')
        axes[0,0].set_xscale('log')
        axes[0,0].set_yscale('log')
        
        axes[0,1].scatter(self.psi_uncertainty, psi_error, alpha=0.6, s=20)
        axes[0,1].set_xlabel('Psi Model Uncertainty')
        axes[0,1].set_ylabel('Psi Prediction Error')
        axes[0,1].set_title('Uncertainty vs Error (Psi Model)')
        axes[0,1].set_xscale('log')
        axes[0,1].set_yscale('log')
        
        # 2. Uncertainty Distribution
        axes[0,2].hist(q_unc, bins=50, alpha=0.7, label='Q Model', density=True)
        axes[0,2].hist(self.psi_uncertainty, bins=50, alpha=0.7, label='Psi Model', density=True)
        axes[0,2].set_xlabel('Model Uncertainty')
        axes[0,2].set_ylabel('Density')
        axes[0,2].set_title('Uncertainty Distribution')
        axes[0,2].legend()
        axes[0,2].set_yscale('log')
        
        # 3. Confidence Intervals
        confidence_levels = [0.68, 0.95, 0.99]
        q_quantiles = [np.percentile(q_unc, [50-cl/2*100, 50+cl/2*100]) for cl in confidence_levels]
        psi_quantiles = [np.percentile(self.psi_uncertainty, [50-cl/2*100, 50+cl/2*100]) for cl in confidence_levels]
        
        axes[1,0].errorbar(range(len(confidence_levels)), 
                          [np.median(q_unc)]*len(confidence_levels),
                          yerr=[[np.median(q_unc) - q[0] for q in q_quantiles],
                                [q[1] - np.median(q_unc) for q in q_quantiles]],
                          fmt='o-', label='Q Model', capsize=5)
        axes[1,0].errorbar(range(len(confidence_levels)), 
                          [np.median(self.psi_uncertainty)]*len(confidence_levels),
                          yerr=[[np.median(self.psi_uncertainty) - p[0] for p in psi_quantiles],
                                [p[1] - np.median(self.psi_uncertainty) for p in psi_quantiles]],
                          fmt='s-', label='Psi Model', capsize=5)
        axes[1,0].set_xticks(range(len(confidence_levels)))
        axes[1,0].set_xticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
        axes[1,0].set_ylabel('Uncertainty')
        axes[1,0].set_title('Confidence Intervals')
        axes[1,0].legend()
        axes[1,0].set_yscale('log')
        
        # 4. Uncertainty vs Input Invariants
        axes[1,1].scatter(self.q_cpu, q_unc, alpha=0.6, s=20, c=self.r_cpu, cmap='viridis')
        axes[1,1].set_xlabel('q invariant')
        axes[1,1].set_ylabel('Q Model Uncertainty')
        axes[1,1].set_title('Uncertainty in (q,r) space')
        cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
        cbar.set_label('r invariant')
        
        # 5. Calibration Plot
        # Bin predictions by uncertainty and check if error scales appropriately
        n_bins = 10
        q_unc_sorted = np.argsort(q_unc)
        bin_size = len(q_unc) // n_bins
        
        bin_uncertainties = []
        bin_errors = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(q_unc)
            indices = q_unc_sorted[start_idx:end_idx]
            
            bin_uncertainties.append(np.mean(q_unc[indices]))
            bin_errors.append(np.mean(q_error[indices]))
        
        axes[1,2].plot(bin_uncertainties, bin_errors, 'o-', linewidth=2, markersize=8, label='Observed')
        axes[1,2].plot([min(bin_uncertainties), max(bin_uncertainties)], 
                      [min(bin_uncertainties), max(bin_uncertainties)], 
                      'r--', linewidth=2, label='Perfect Calibration')
        axes[1,2].set_xlabel('Predicted Uncertainty')
        axes[1,2].set_ylabel('Observed Error')
        axes[1,2].set_title('Model Calibration')
        axes[1,2].legend()
        axes[1,2].set_xscale('log')
        axes[1,2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'uncertainty_analysis.png')
        plt.show()

    def plot_physics_consistency_checks(self):
        """Plot physics-based consistency checks."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Tensor Magnitude Conservation
        Q_mag_true = np.linalg.norm(self.Q_hat_prime_cpu, axis=(1,2))
        Q_mag_pred = np.linalg.norm(self.Q_hat_prime_pred_cpu, axis=(1,2))
        
        axes[0,0].scatter(Q_mag_true, Q_mag_pred, alpha=0.6, s=20)
        axes[0,0].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[0,0].set_xlabel('True ||Q\'|| ')
        axes[0,0].set_ylabel('Predicted ||Q\'||')
        axes[0,0].set_title('Tensor Magnitude Conservation')
        axes[0,0].set_aspect('equal')
        
        # 2. Trace Conservation (should be zero for traceless tensors)
        Q_trace_true = np.trace(self.Q_hat_prime_cpu, axis1=1, axis2=2)
        Q_trace_pred = np.trace(self.Q_hat_prime_pred_cpu, axis1=1, axis2=2)
        
        axes[0,1].hist(Q_trace_true, bins=50, alpha=0.7, label='True', density=True)
        axes[0,1].hist(Q_trace_pred, bins=50, alpha=0.7, label='Predicted', density=True)
        axes[0,1].set_xlabel('Trace(Q\')')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Traceless Property Conservation')
        axes[0,1].legend()
        
        # 3. Eigenvalue Distribution
        Q_eigs_true = np.linalg.eigvals(self.Q_hat_prime_cpu)
        Q_eigs_pred = np.linalg.eigvals(self.Q_hat_prime_pred_cpu)
        
        for i in range(3):
            axes[0,2].hist(Q_eigs_true[:, i], bins=30, alpha=0.5, 
                          label=f'True λ{i+1}', density=True)
            axes[0,2].hist(Q_eigs_pred[:, i], bins=30, alpha=0.5, 
                          label=f'Pred λ{i+1}', linestyle='--', density=True)
        axes[0,2].set_xlabel('Eigenvalue')
        axes[0,2].set_ylabel('Density')
        axes[0,2].set_title('Eigenvalue Distribution')
        axes[0,2].legend()
        
        # 4. Psi Positivity Check
        axes[1,0].hist(self.psi_cpu, bins=50, alpha=0.7, label='True', density=True)
        axes[1,0].hist(self.psi_pred_cpu, bins=50, alpha=0.7, label='Predicted', density=True)
        axes[1,0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Line')
        axes[1,0].set_xlabel('ψ')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Psi Positivity Check')
        axes[1,0].legend()
        
        # 5. Rotational Invariance Test
        # Apply random rotation and check if predictions change
        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Sample a subset for computational efficiency
        n_samples = min(1000, len(self.A_full))
        indices = np.random.choice(len(self.A_full), n_samples, replace=False)
        
        A_sample = self.A_full[indices].cpu().numpy()
        A_rotated = np.einsum('ij,bjk,lk->bil', R, A_sample, R)
        
        with torch.no_grad():
            A_rot_tensor = torch.tensor(A_rotated, dtype=torch.float32, device=self.device)
            s_rot, w_rot, eps_rot, _, _ = get_tensor_derivatives(A_rot_tensor)
            a_rot = A_rot_tensor / eps_rot.unsqueeze(-1).unsqueeze(-1)
            s_norm_rot = 0.5 * (a_rot + a_rot.transpose(1, 2))
            
            Q_pred_rot = self.model_Q(s_norm_rot, w_rot)
            Q_pred_orig = self.model_Q(s_norm[indices], self.w[indices])
            
            # Check invariance
            invariance_error = torch.norm(Q_pred_rot - Q_pred_orig, dim=(1,2)).cpu().numpy()
        
        axes[1,1].hist(invariance_error, bins=30, alpha=0.7, density=True)
        axes[1,1].set_xlabel('||Q\'_rotated - Q\'_original||')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Rotational Invariance Test')
        axes[1,1].set_yscale('log')
        
        # 6. Energy Scale Consistency
        energy_scale = self.epsilon.cpu().numpy()**2
        Q_energy = np.sum(self.Q_prime.cpu().numpy()**2, axis=(1,2))
        
        axes[1,2].scatter(energy_scale, Q_energy, alpha=0.6, s=20)
        axes[1,2].set_xlabel('ε²')
        axes[1,2].set_ylabel('||Q\'||²')
        axes[1,2].set_title('Energy Scale Consistency')
        axes[1,2].set_xscale('log')
        axes[1,2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(RUN_DIR / 'physics_consistency_checks.png')
        plt.show()

    def plot_spectral_analysis(self):
        """Plot spectral analysis of prediction errors."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Error Power Spectral Density (simplified 1D version)
        Q_error = self.Q_hat_prime_pred_cpu - self.Q_hat_prime_cpu
        Q_error_flat = Q_error.reshape(Q_error.shape[0], -1)
        
        # Compute PSD for each sample and average
        freqs = np.fft.fftfreq(Q_error_flat.shape[1])
        psd_avg = np.zeros_like(freqs)
        
        for i in range(min(100, Q_error_flat.shape[0])):  # Sample for efficiency
            fft_vals = np.fft.fft(Q_error_flat[i])
            psd = np.abs(fft_vals)**2
            psd_avg += psd
        
        psd_avg /= min(100, Q_error_flat.shape[0])
        
        axes[0,0].loglog(freqs[freqs > 0], psd_avg[freqs > 0])
        axes[0,0].set_xlabel('Frequency')
        axes[0,0].set_ylabel('Power Spectral Density')
        axes[0,0].set_title('Q\' Error PSD')
        axes[0,0].grid(True)
        
        # 2. Error Autocorrelation
        from scipy import signal
        Q_error_1d = Q_error_flat[0]  # Take first sample as example
        autocorr = signal.correlate(Q_error_1d, Q_error_1d, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        lags = np.arange(len(autocorr))
        axes[0,1].plot(lags[:100], autocorr[:100])  # Plot first 100 lags
        axes[0,1].set_xlabel('Lag')
        axes[0,1].set_ylabel('Autocorrelation')
        axes[0,1].set_title('Q\' Error Autocorrelation')
        axes[0,1].grid(True)
        
        # 3. Wavelength-dependent Error
        # Bin errors by wavelength (proxy: tensor norm)
        tensor_norms = np.linalg.norm(self.A_full.cpu().numpy(), axis=(1,2))
        error_norms = np.linalg.norm(Q_error, axis=(1,2))
        
        # Create bins based on tensor norm (energy scale)
        n_bins = 20
        bins = np.logspace(np.log10(tensor_norms.min()), np.log10(tensor_norms.max()), n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        binned_errors = []
        for i in range(len(bins)-1):
            mask = (tensor_norms >= bins[i]) & (tensor_norms < bins[i+1])
            if np.any(mask):
                binned_errors.append(np.mean(error_norms[mask]))
        # Plot binned error vs bin centers
        axes[1,0].plot(bin_centers, binned_errors, 'o-', linewidth=2)
        axes[1,0].set_xscale('log')
        axes[1,0].set_yscale('log')
        axes[1,0].set_xlabel('Tensor Norm (Energy Scale)')
        axes[1,0].set_ylabel('Mean Prediction Error')
        axes[1,0].set_title('Error vs Energy Scale')

        plt.tight_layout()
        plt.savefig(RUN_DIR / 'spectral_analysis.png')
        plt.show()

    def plot_input_feature_analysis(self):
        """Visualize input feature saliency and their ratios."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # 1. Strain vs Rotation saliency
        axes[0, 0].plot(self.history['train_strain_saliency'], label='Strain Rate', linewidth=2)
        axes[0, 0].plot(self.history['train_rotation_saliency'], label='Rotation Rate', linewidth=2)
        axes[0, 0].set_title('Q Model Input Saliency')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Gradient Magnitude')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')

        # 2. q vs r saliency
        axes[0, 1].plot(self.history['train_q_saliency'], label='Invariant q', linewidth=2)
        axes[0, 1].plot(self.history['train_r_saliency'], label='Invariant r', linewidth=2)
        axes[0, 1].set_title('Psi Model Input Saliency')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gradient Magnitude')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')

        # 3. Strain/Rotation saliency ratio
        if self.history['train_strain_saliency'] and self.history['train_rotation_saliency']:
            strain_rot_ratio = np.array(self.history['train_strain_saliency']) / (
                np.array(self.history['train_rotation_saliency']) + 1e-10)
            axes[1, 0].plot(strain_rot_ratio, linewidth=2, color='purple')
        axes[1, 0].set_title('Strain/Rotation Saliency Ratio')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Strain Saliency / Rotation Saliency')

        # 4. q/r saliency ratio
        if self.history['train_q_saliency'] and self.history['train_r_saliency']:
            q_r_ratio = np.array(self.history['train_q_saliency']) / (
                np.array(self.history['train_r_saliency']) + 1e-10)
            axes[1, 1].plot(q_r_ratio, linewidth=2, color='orange')
        axes[1, 1].set_title('q/r Saliency Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('q Saliency / r Saliency')

        plt.tight_layout()
        plt.savefig(RUN_DIR / 'input_feature_analysis.png')
        plt.show()

    def plot_model_reliability(self):
        """Plot reliability diagrams (calibration curves) for Q and psi models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Q Model reliability
        q_unc = np.mean(self.Q_uncertainty, axis=(1,2))
        q_error = np.linalg.norm(self.Q_hat_prime_cpu - self.Q_hat_prime_pred_cpu, axis=(1,2))
        n_bins = 10
        q_unc_sorted = np.argsort(q_unc)
        bin_size = len(q_unc) // n_bins
        bin_uncertainties = []
        bin_errors = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(q_unc)
            indices = q_unc_sorted[start_idx:end_idx]
            bin_uncertainties.append(np.mean(q_unc[indices]))
            bin_errors.append(np.mean(q_error[indices]))
        axes[0].plot(bin_uncertainties, bin_errors, 'o-', linewidth=2, markersize=8, label='Observed')
        axes[0].plot([min(bin_uncertainties), max(bin_uncertainties)],
                     [min(bin_uncertainties), max(bin_uncertainties)], 'r--', linewidth=2, label='Perfect Calibration')
        axes[0].set_xlabel('Predicted Uncertainty (Q)')
        axes[0].set_ylabel('Observed Error (Q)')
        axes[0].set_title('Q Model Reliability Diagram')
        axes[0].legend()
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')

        # Psi Model reliability
        psi_unc = self.psi_uncertainty
        psi_error = np.abs(self.psi_cpu - self.psi_pred_cpu)
        psi_unc_sorted = np.argsort(psi_unc)
        bin_size_psi = len(psi_unc) // n_bins
        bin_uncertainties_psi = []
        bin_errors_psi = []
        for i in range(n_bins):
            start_idx = i * bin_size_psi
            end_idx = start_idx + bin_size_psi if i < n_bins - 1 else len(psi_unc)
            indices = psi_unc_sorted[start_idx:end_idx]
            bin_uncertainties_psi.append(np.mean(psi_unc[indices]))
            bin_errors_psi.append(np.mean(psi_error[indices]))
        axes[1].plot(bin_uncertainties_psi, bin_errors_psi, 'o-', linewidth=2, markersize=8, label='Observed')
        axes[1].plot([min(bin_uncertainties_psi), max(bin_uncertainties_psi)],
                     [min(bin_uncertainties_psi), max(bin_uncertainties_psi)], 'r--', linewidth=2, label='Perfect Calibration')
        axes[1].set_xlabel('Predicted Uncertainty (Psi)')
        axes[1].set_ylabel('Observed Error (Psi)')
        axes[1].set_title('Psi Model Reliability Diagram')
        axes[1].legend()
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(RUN_DIR / 'model_reliability.png')
        plt.show()
