import os
import random
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class AugmentedInvariantBasedModel:
    """
    Augmented Invariant-Based Model (AIBM) for modeling the pressure Hessian tensor
    as described in the paper "An augmented invariant-based model of the pressure
    Hessian tensor using a combination of physics-assisted neural networks".
    """
    def __init__(self, device='cuda', config=None):
        """
        Initialize the AIBM model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            config: Configuration dictionary for model parameters
        """
        self.device = device
        
        # Default configuration
        self.config = {
            # TBNN Alignment Network
            'tbnn_input_dim': 5,           # Number of tensor invariants (λ1...λ5)
            'tbnn_hidden_layers': [50, 100, 100, 100, 50],
            'tbnn_output_coeffs': 10,      # Number of tensor basis coefficients g(n)
            'tbnn_batch_size': 16384,      # From paper
            'tbnn_lr': 1e-3,               # From paper
            'tbnn_dropout': 0.1,           # From paper
            
            # Magnitude Network
            'mag_input_dim': 2,            # q and r invariants
            'mag_hidden_layers': [50, 80, 50],
            'mag_batch_size': 4096,        # From paper
            'mag_lr': 5e-3,                # From paper
            'mag_dropout': 0.1,            # From paper
            
            # Training
            'epochs': 400,                 # From paper
            'early_stopping_patience': 10, # From paper
        }
        
        # Update with user configuration if provided
        if config:
            self.config.update(config)
        
        # Initialize models
        self.tbnn_model = TensorBasisNeuralNetwork(
            input_dim=self.config['tbnn_input_dim'],
            hidden_layers=self.config['tbnn_hidden_layers'],
            output_coeffs=self.config['tbnn_output_coeffs'],
            dropout=self.config['tbnn_dropout']
        ).to(device)
        
        self.magnitude_model = MagnitudePredictionNetwork(
            input_dim=self.config['mag_input_dim'],
            hidden_layers=self.config['mag_hidden_layers'],
            dropout=self.config['mag_dropout']
        ).to(device)
        
        # Initialize optimizers
        self.tbnn_optimizer = optim.Adamax(
            self.tbnn_model.parameters(), 
            lr=self.config['tbnn_lr']
        )
        
        self.magnitude_optimizer = optim.Adamax(
            self.magnitude_model.parameters(), 
            lr=self.config['mag_lr']
        )
    
    def train(self, train_data, val_data=None):
        """
        Train both neural networks of the AIBM model.
        
        Args:
            train_data: Dictionary containing training data tensors
            val_data: Dictionary containing validation data tensors (optional)
        """
        print("Training TBNN Alignment Network...")
        self._train_tbnn(train_data, val_data)
        
        print("\nTraining Magnitude Prediction Network...")
        self._train_magnitude(train_data, val_data)
    
    def _train_tbnn(self, train_data, val_data=None):
        """
        Train the Tensor Basis Neural Network for alignment prediction.
        """
        # Create data loaders
        train_dataset = TensorDataset(
            train_data['lambda_invariants'],
            train_data['s_norm'],
            train_data['w_norm'],
            train_data['Q_tilde_prime']
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['tbnn_batch_size'],
            shuffle=True,
            pin_memory=True
        )
        
        if val_data:
            val_dataset = TensorDataset(
                val_data['lambda_invariants'],
                val_data['s_norm'],
                val_data['w_norm'],
                val_data['Q_tilde_prime']
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['tbnn_batch_size'],
                shuffle=False,
                pin_memory=True
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['epochs']):
            # Training
            self.tbnn_model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", leave=False):
                lambda_invariants, s_norm, w_norm, Q_tilde_prime_dns = [b.to(self.device) for b in batch]
                
                self.tbnn_optimizer.zero_grad()
                
                # Forward pass
                g_coeffs = self.tbnn_model(lambda_invariants)
                Q_tilde_prime_pred = self.tbnn_model.construct_Q_tilde_prime(g_coeffs, s_norm, w_norm)
                
                # Compute loss
                loss = alignment_loss(Q_tilde_prime_pred, s_norm, Q_tilde_prime_dns)
                
                # Backward pass and optimize
                loss.backward()
                self.tbnn_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_data:
                val_loss = self._validate_tbnn(val_loader)
                val_losses.append(val_loss)
                
                print(f"Epoch [{epoch+1}/{self.config['epochs']}], "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.tbnn_model.state_dict(), 'best_tbnn_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Load best model
                        self.tbnn_model.load_state_dict(torch.load('best_tbnn_model.pt'))
                        break
            else:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}], Train Loss: {train_loss:.6f}")
        
        return train_losses, val_losses
    
    def _validate_tbnn(self, val_loader):
        """
        Validate the Tensor Basis Neural Network.
        """
        self.tbnn_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                lambda_invariants, s_norm, w_norm, Q_tilde_prime_dns = [b.to(self.device) for b in batch]
                
                # Forward pass
                g_coeffs = self.tbnn_model(lambda_invariants)
                Q_tilde_prime_pred = self.tbnn_model.construct_Q_tilde_prime(g_coeffs, s_norm, w_norm)
                
                # Compute loss
                loss = alignment_loss(Q_tilde_prime_pred, s_norm, Q_tilde_prime_dns)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _train_magnitude(self, train_data, val_data=None):
        """
        Train the Magnitude Prediction Network.
        """
        # Create data loaders
        train_dataset = TensorDataset(
            train_data['q_r_invariants'],
            train_data['psi']
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['mag_batch_size'],
            shuffle=True,
            pin_memory=True
        )
        
        if val_data:
            val_dataset = TensorDataset(
                val_data['q_r_invariants'],
                val_data['psi']
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['mag_batch_size'],
                shuffle=False,
                pin_memory=True
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['epochs']):
            # Training
            self.magnitude_model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", leave=False):
                q_r_invariants, psi_dns = [b.to(self.device) for b in batch]
                
                self.magnitude_optimizer.zero_grad()
                
                # Forward pass
                psi_pred = self.magnitude_model(q_r_invariants)
                
                # Compute loss
                loss = magnitude_loss(psi_pred, psi_dns)
                
                # Backward pass and optimize
                loss.backward()
                self.magnitude_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_data:
                val_loss = self._validate_magnitude(val_loader)
                val_losses.append(val_loss)
                
                print(f"Epoch [{epoch+1}/{self.config['epochs']}], "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.magnitude_model.state_dict(), 'best_magnitude_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Load best model
                        self.magnitude_model.load_state_dict(torch.load('best_magnitude_model.pt'))
                        break
            else:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}], Train Loss: {train_loss:.6f}")
        
        return train_losses, val_losses
    
    def _validate_magnitude(self, val_loader):
        """
        Validate the Magnitude Prediction Network.
        """
        self.magnitude_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                q_r_invariants, psi_dns = [b.to(self.device) for b in batch]
                
                # Forward pass
                psi_pred = self.magnitude_model(q_r_invariants)
                
                # Compute loss
                loss = magnitude_loss(psi_pred, psi_dns)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def predict(self, A_tensor):
        """
        Make a prediction using the trained AIBM model.
        
        Args:
            A_tensor: Velocity gradient tensor (batch_size, 3, 3)
            
        Returns:
            Q_prime: Predicted anisotropic pressure Hessian tensor
            Q_tilde_prime: Normalized anisotropic pressure Hessian tensor
            psi: Predicted magnitude
        """
        self.tbnn_model.eval()
        self.magnitude_model.eval()
        
        with torch.no_grad():
            # Normalize A tensor
            A_norm, epsilon_sq = normalize_tensor(A_tensor)
            
            # Calculate s_norm and w_norm
            s_norm, w_norm = calculate_s_w(A_norm)
            
            # Calculate invariants
            lambda_invariants = calculate_lambda_invariants(s_norm, w_norm)
            q_r_invariants = calculate_q_r_invariants(s_norm, w_norm)
            
            # Predict alignment (Q_tilde_prime)
            g_coeffs = self.tbnn_model(lambda_invariants)
            Q_tilde_prime = self.tbnn_model.construct_Q_tilde_prime(g_coeffs, s_norm, w_norm)
            
            # Predict magnitude (psi)
            psi = self.magnitude_model(q_r_invariants)
            
            # Combine to get Q_prime
            Q_prime = psi.unsqueeze(-1).unsqueeze(-1) * Q_tilde_prime
            
        return Q_prime, Q_tilde_prime, psi
    
    def evaluate(self, test_data):
        """
        Evaluate the AIBM model on test data.
        
        Args:
            test_data: Dictionary containing test data tensors
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.tbnn_model.eval()
        self.magnitude_model.eval()
        
        A_tensor = test_data['A_tensor'].to(self.device)
        Q_prime_dns = test_data['Q_prime'].to(self.device)
        Q_tilde_prime_dns = test_data['Q_tilde_prime'].to(self.device)
        psi_dns = test_data['psi'].to(self.device)
        s_norm_dns = test_data['s_norm'].to(self.device)
        
        Q_prime_pred, Q_tilde_prime_pred, psi_pred = self.predict(A_tensor)
        
        # Calculate losses
        alignment_loss_val = alignment_loss(Q_tilde_prime_pred, s_norm_dns, Q_tilde_prime_dns)
        magnitude_loss_val = magnitude_loss(psi_pred, psi_dns)
        
        # Calculate RMSE
        rmse = torch.sqrt(torch.mean((Q_prime_pred - Q_prime_dns)**2))
        
        return {
            'alignment_loss': alignment_loss_val.item(),
            'magnitude_loss': magnitude_loss_val.item(),
            'rmse': rmse.item(),
            'Q_prime_pred': Q_prime_pred.cpu(),
            'Q_tilde_prime_pred': Q_tilde_prime_pred.cpu(),
            'psi_pred': psi_pred.cpu()
        }


class TensorBasisNeuralNetwork(nn.Module):
    """
    Tensor Basis Neural Network for predicting alignment of 
    pressure Hessian tensor eigenvectors.
    """
    def __init__(self, input_dim=5, hidden_layers=[50, 100, 100, 100, 50], 
                 output_coeffs=10, dropout=0.1):
        """
        Initialize the TBNN.
        
        Args:
            input_dim: Number of tensor invariants (λ1...λ5)
            hidden_layers: List of hidden layer dimensions
            output_coeffs: Number of tensor basis coefficients
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Output layer for coefficients
        self.output_layer = nn.Linear(current_dim, output_coeffs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass to predict coefficients g(n).
        
        Args:
            x: Tensor invariants (batch_size, 5)
        
        Returns:
            g_coeffs: Tensor basis coefficients (batch_size, 10)
        """
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.1)
            x = self.dropout(x)
        
        g_coeffs = self.output_layer(x)
        return g_coeffs
    
    def construct_Q_tilde_prime(self, g_coeffs, s_norm, w_norm):
        """
        Construct Q_tilde_prime from coefficients and tensor bases.
        
        Args:
            g_coeffs: Tensor basis coefficients (batch_size, 10)
            s_norm: Normalized strain-rate tensor (batch_size, 3, 3)
            w_norm: Normalized rotation-rate tensor (batch_size, 3, 3)
        
        Returns:
            Q_tilde_prime: Normalized anisotropic pressure Hessian tensor
        """
        # Get tensor bases
        tensor_bases = calculate_tensor_bases(s_norm, w_norm)
        
        # Q_hat_prime = sum_{n=1}^{10} g^(n) * T^(n)
        Q_hat_prime = torch.sum(
            g_coeffs.unsqueeze(-1).unsqueeze(-1) * tensor_bases,
            dim=1
        )
        
        # Normalize to get Q_tilde_prime
        Q_tilde_prime = normalize_tensor_preserving_shape(Q_hat_prime)
        
        return Q_tilde_prime


class MagnitudePredictionNetwork(nn.Module):
    """
    Neural Network for predicting the magnitude (psi) of
    the anisotropic pressure Hessian tensor.
    """
    def __init__(self, input_dim=2, hidden_layers=[50, 80, 50], dropout=0.1):
        """
        Initialize the Magnitude Prediction Network.
        
        Args:
            input_dim: Number of invariants (q and r)
            hidden_layers: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Output layer for psi
        self.output_layer = nn.Linear(current_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass to predict psi.
        
        Args:
            x: q and r invariants (batch_size, 2)
        
        Returns:
            psi: Magnitude of anisotropic pressure Hessian tensor (batch_size, 1)
        """
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.1)
            x = self.dropout(x)
        
        psi = self.output_layer(x)
        return psi


class TensorDataset(Dataset):
    """
    Dataset class for tensor data.
    """
    def __init__(self, *tensors):
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)


# --- Tensor Operations and Loss Functions ---

def normalize_tensor(tensor):
    """
    Normalize the velocity gradient tensor A. (Eq. 13)
    
    Args:
        tensor: Input tensor of shape (batch_size, 3, 3)
    
    Returns:
        normalized_tensor: Normalized tensor
        epsilon_sq: Square of the normalization factor
    """
    # Calculate epsilon_sq = A_mn A_mn
    epsilon_sq = torch.sum(tensor * tensor, dim=(1, 2), keepdim=True)
    epsilon = torch.sqrt(epsilon_sq + 1e-12)
    
    # Normalize
    normalized_tensor = tensor / epsilon
    
    return normalized_tensor, epsilon_sq.squeeze(-1).squeeze(-1)


def normalize_tensor_preserving_shape(tensor):
    """
    Self-normalize a tensor preserving its shape. (Eq. 10)
    
    Args:
        tensor: Input tensor of any shape
    
    Returns:
        normalized_tensor: Normalized tensor with the same shape
    """
    # Calculate Frobenius norm
    norm = torch.sqrt(torch.sum(tensor * tensor, dim=(-2, -1), keepdim=True) + 1e-12)
    
    # Normalize
    normalized_tensor = tensor / norm
    
    return normalized_tensor


def calculate_s_w(A_norm):
    """
    Calculate normalized strain-rate (s) and rotation-rate (w) tensors. (Eq. 14)
    
    Args:
        A_norm: Normalized velocity gradient tensor (batch_size, 3, 3)
    
    Returns:
        s_norm: Normalized strain-rate tensor (batch_size, 3, 3)
        w_norm: Normalized rotation-rate tensor (batch_size, 3, 3)
    """
    # s = (A + A^T) / 2, symmetric part
    s_norm = 0.5 * (A_norm + A_norm.transpose(1, 2))
    
    # w = (A - A^T) / 2, anti-symmetric part
    w_norm = 0.5 * (A_norm - A_norm.transpose(1, 2))
    
    return s_norm, w_norm


def calculate_lambda_invariants(s_norm, w_norm):
    """
    Calculate the five lambda invariants. (Eq. 17)
    
    Args:
        s_norm: Normalized strain-rate tensor (batch_size, 3, 3)
        w_norm: Normalized rotation-rate tensor (batch_size, 3, 3)
    
    Returns:
        lambda_invariants: Five invariants (batch_size, 5)
    """
    # Matrix products (batched)
    s_sq = torch.bmm(s_norm, s_norm)
    w_sq = torch.bmm(w_norm, w_norm)
    s_cube = torch.bmm(s_sq, s_norm)
    w_sq_s = torch.bmm(w_sq, s_norm)
    w_sq_s_sq = torch.bmm(w_sq, s_sq)
    
    # Calculate invariants
    lambda1 = torch.sum(s_norm * s_norm, dim=(1, 2))  # Tr(s^2)
    lambda2 = torch.sum(w_norm * w_norm, dim=(1, 2))  # Tr(w^2)
    lambda3 = torch.diagonal(s_cube, dim1=1, dim2=2).sum(dim=1)  # Tr(s^3)
    lambda4 = torch.diagonal(w_sq_s, dim1=1, dim2=2).sum(dim=1)  # Tr(w^2 s)
    lambda5 = torch.diagonal(w_sq_s_sq, dim1=1, dim2=2).sum(dim=1)  # Tr(w^2 s^2)
    
    return torch.stack([lambda1, lambda2, lambda3, lambda4, lambda5], dim=1)


def calculate_q_r_invariants(s_norm, w_norm):
    """
    Calculate q and r invariants from normalized strain and rotation rate tensors. (Eq. 21)
    For incompressible flow, p = 0.
    
    Args:
        s_norm: Normalized strain-rate tensor (batch_size, 3, 3)
        w_norm: Normalized rotation-rate tensor (batch_size, 3, 3)
    
    Returns:
        q_r_invariants: q and r invariants (batch_size, 2)
    """
    # Calculate matrix products
    s_sq = torch.bmm(s_norm, s_norm)
    w_sq = torch.bmm(w_norm, w_norm)
    s_cube = torch.bmm(s_sq, s_norm)
    w_sq_s = torch.bmm(w_sq, s_norm)
    
    # Calculate traces
    tr_s_sq = torch.diagonal(s_sq, dim1=1, dim2=2).sum(dim=1)
    tr_w_sq = torch.diagonal(w_sq, dim1=1, dim2=2).sum(dim=1)
    tr_s_cube = torch.diagonal(s_cube, dim1=1, dim2=2).sum(dim=1)
    tr_w_sq_s = torch.diagonal(w_sq_s, dim1=1, dim2=2).sum(dim=1)
    
    # Calculate q and r (p = 0 for incompressible flow)
    q = -0.5 * (tr_s_sq + tr_w_sq)
    r = -(1.0/3.0) * (tr_s_cube + 3.0 * tr_w_sq_s)
    
    return torch.stack([q, r], dim=1)


def calculate_tensor_bases(s_norm, w_norm):
    """
    Calculate the 10 anisotropic tensor bases T^(n). (Eq. 16)
    
    Args:
        s_norm: Normalized strain-rate tensor (batch_size, 3, 3)
        w_norm: Normalized rotation-rate tensor (batch_size, 3, 3)
    
    Returns:
        tensor_bases: Ten tensor bases (batch_size, 10, 3, 3)
    """
    batch_size = s_norm.shape[0]
    device = s_norm.device
    
    # Create identity tensor
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Pre-calculate matrix products
    s_sq = torch.bmm(s_norm, s_norm)
    w_sq = torch.bmm(w_norm, w_norm)
    s_w = torch.bmm(s_norm, w_norm)
    w_s = torch.bmm(w_norm, s_norm)
    s_sq_w = torch.bmm(s_sq, w_norm)
    w_s_sq = torch.bmm(w_norm, s_sq)
    w_sq_s = torch.bmm(w_sq, s_norm)
    s_w_sq = torch.bmm(s_norm, w_sq)
    w_s_w_sq = torch.bmm(w_norm, torch.bmm(s_norm, w_sq))
    w_sq_s_w = torch.bmm(w_sq, torch.bmm(s_norm, w_norm))
    s_w_s_sq = torch.bmm(s_norm, torch.bmm(w_norm, s_sq))
    s_sq_w_s = torch.bmm(s_sq, torch.bmm(w_norm, s_norm))
    s_sq_w_sq = torch.bmm(s_sq, w_sq)
    w_sq_s_sq = torch.bmm(w_sq, s_sq)
    w_s_sq_w_sq = torch.bmm(w_norm, torch.bmm(s_sq, w_sq))
    w_sq_s_sq_w = torch.bmm(w_sq, torch.bmm(s_sq, w_norm))
    
    # Calculate traces needed for deviatoric parts
    tr_s_sq = torch.diagonal(s_sq, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    tr_w_sq = torch.diagonal(w_sq, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    tr_s_w_sq = torch.diagonal(s_w_sq, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    tr_s_sq_w_sq = torch.diagonal(s_sq_w_sq, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    
    # Initialize tensor bases
    tensor_bases = torch.zeros(batch_size, 10, 3, 3, device=device)
    
    # T(1) = s
    tensor_bases[:, 0] = s_norm
    
    # T(2) = sw - ws
    tensor_bases[:, 1] = s_w - w_s
    
    # T(3) = s^2 - (1/3)*I*Tr(s^2)
    tensor_bases[:, 2] = s_sq - (1.0/3.0) * I * tr_s_sq
    
    # T(4) = w^2 - (1/3)*I*Tr(w^2)
    tensor_bases[:, 3] = w_sq - (1.0/3.0) * I * tr_w_sq
    
    # T(5) = ws^2 - s^2w
    tensor_bases[:, 4] = w_s_sq - s_sq_w
    
    # T(6) = w^2s + sw^2 - (2/3)*I*Tr(sw^2)
    tensor_bases[:, 5] = w_sq_s + s_w_sq - (2.0/3.0) * I * tr_s_w_sq
    
    # T(7) = wsw^2 - w^2sw
    tensor_bases[:, 6] = w_s_w_sq - w_sq_s_w
    
    # T(8) = sws^2 - s^2ws
    tensor_bases[:, 7] = s_w_s_sq - s_sq_w_s
    
    # T(9) = w^2s^2 + s^2w^2 - (2/3)*I*Tr(s^2w^2)
    tensor_bases[:, 8] = w_sq_s_sq + s_sq_w_sq - (2.0/3.0) * I * tr_s_sq_w_sq
    
    # T(10) = ws^2w^2 - w^2s^2w
    tensor_bases[:, 9] = w_s_sq_w_sq - w_sq_s_sq_w
    
    return tensor_bases


def calculate_vorticity(w_norm):
    """
    Calculate the vorticity vector from the rotation rate tensor. (Eq. 30)
    
    Args:
        w_norm: Normalized rotation-rate tensor (batch_size, 3, 3)
    
    Returns:
        vorticity: Vorticity vector (batch_size, 3)
    """
    batch_size = w_norm.shape[0]
    device = w_norm.device
    vorticity = torch.zeros(batch_size, 3, device=device)
    
    # Vorticity components using Levi-Civita tensor
    vorticity[:, 0] = 2 * w_norm[:, 2, 1]  # x-component = 2*w_23 = -2*w_32
    vorticity[:, 1] = 2 * w_norm[:, 0, 2]  # y-component = 2*w_31 = -2*w_13
    vorticity[:, 2] = 2 * w_norm[:, 1, 0]  # z-component = 2*w_12 = -2*w_21
    
    return vorticity


def calculate_eigensystem(tensor_batch):
    """
    Calculate eigenvalues and eigenvectors of a batch of 3x3 symmetric tensors.
    Handles numerical issues and sorts eigenvalues/vectors in descending order.
    
    Args:
        tensor_batch: Batch of tensors (batch_size, 3, 3)
    
    Returns:
        eigenvalues: Sorted eigenvalues in descending order (batch_size, 3)
        eigenvectors: Corresponding eigenvectors (batch_size, 3, 3)
    """
    batch_size = tensor_batch.shape[0]
    device = tensor_batch.device
    
    # Initialize outputs
    eigenvalues = torch.zeros(batch_size, 3, device=device)
    eigenvectors = torch.zeros(batch_size, 3, 3, device=device)
    
    # Process each tensor in batch
    for i in range(batch_size):
        try:
            # torch.linalg.eigh is more stable for symmetric tensors
            # Make sure tensor is symmetric to avoid numerical issues
            tensor_sym = 0.5 * (tensor_batch[i] + tensor_batch[i].transpose(0, 1))
            e_vals, e_vecs = torch.linalg.eigh(tensor_sym)
            
            # Sort eigenvalues in descending order and reorder eigenvectors
            sorted_indices = torch.argsort(e_vals, descending=True)
            eigenvalues[i] = e_vals[sorted_indices]
            eigenvectors[i] = e_vecs[:, sorted_indices]
            
        except torch.linalg.LinAlgError:
            # Fallback for numerical issues
            eigenvalues[i] = torch.tensor([1.0, 0.0, -1.0], device=device)
            eigenvectors[i] = torch.eye(3, device=device)
    
    return eigenvalues, eigenvectors


def alignment_loss(Q_tilde_prime_pred, s_norm, Q_tilde_prime_dns):
    """
    Calculate alignment loss based on Euler angles (Eq. 20).
    
    Args:
        Q_tilde_prime_pred: Predicted normalized pressure Hessian tensor (batch_size, 3, 3)
        s_norm: Normalized strain-rate tensor (batch_size, 3, 3)
        Q_tilde_prime_dns: Ground truth normalized pressure Hessian tensor (batch_size, 3, 3)
    
    Returns:
        loss: Alignment loss (scalar)
    """
    batch_size = Q_tilde_prime_pred.shape[0]
    device = Q_tilde_prime_pred.device
    
    # Calculate eigenvalues and eigenvectors of s_norm
    _, s_eigenvectors = calculate_eigensystem(s_norm)
    
    # Extract eigenvectors (ordered as most extensive, intermediate, most compressive)
    e_s_alpha = s_eigenvectors[:, :, 0]  # Most extensive strain eigenvector
    e_s_beta = s_eigenvectors[:, :, 1]   # Intermediate strain eigenvector
    e_s_gamma = s_eigenvectors[:, :, 2]  # Most compressive strain eigenvector
    
    # Calculate eigenvalues and eigenvectors of predicted Q_tilde_prime
    _, Q_pred_eigenvectors = calculate_eigensystem(Q_tilde_prime_pred)
    
    # Extract eigenvectors
    e_q_alpha_pred = Q_pred_eigenvectors[:, :, 0]  # Largest eigenvalue eigenvector
    e_q_beta_pred = Q_pred_eigenvectors[:, :, 1]   # Intermediate eigenvalue eigenvector
    e_q_gamma_pred = Q_pred_eigenvectors[:, :, 2]  # Smallest eigenvalue eigenvector
    
    # Calculate eigenvalues and eigenvectors of DNS Q_tilde_prime
    _, Q_dns_eigenvectors = calculate_eigensystem(Q_tilde_prime_dns)
    
    # Extract eigenvectors
    e_q_alpha_dns = Q_dns_eigenvectors[:, :, 0]
    e_q_beta_dns = Q_dns_eigenvectors[:, :, 1]
    e_q_gamma_dns = Q_dns_eigenvectors[:, :, 2]
    
    # Calculate alignment angles for predictions
    # Calculate cos(xi) - angle between e_q_gamma and e_s_gamma
    cos_xi_pred = torch.sum(e_q_gamma_pred * e_s_gamma, dim=1)
    
    # Calculate e_proj for theta angle - Eq. 19
    e_proj_prime_pred = e_q_alpha_pred - torch.sum(e_q_alpha_pred * e_s_gamma.unsqueeze(2), dim=1).unsqueeze(1) * e_s_gamma
    e_proj_norm_pred = torch.norm(e_proj_prime_pred, dim=1, keepdim=True)
    e_proj_pred = e_proj_prime_pred / (e_proj_norm_pred + 1e-8)
    
    # Calculate cos(theta) - angle between e_s_alpha and e_proj
    cos_theta_pred = torch.sum(e_s_alpha * e_proj_pred, dim=1)
    
    # Calculate e_norm for eta angle - Eq. 19
    e_norm_pred = torch.cross(e_proj_pred, e_s_gamma, dim=1)
    e_norm_norm_pred = torch.norm(e_norm_pred, dim=1, keepdim=True)
    e_norm_pred = e_norm_pred / (e_norm_norm_pred + 1e-8)
    
    # Calculate cos(eta) - angle between e_q_beta and e_norm
    cos_eta_pred = torch.sum(e_q_beta_pred * e_norm_pred, dim=1)
    
    # Calculate alignment angles for DNS data (ground truth)
    # Calculate cos(xi) - angle between e_q_gamma and e_s_gamma
    cos_xi_dns = torch.sum(e_q_gamma_dns * e_s_gamma, dim=1)
    
    # Calculate e_proj for theta angle
    e_proj_prime_dns = e_q_alpha_dns - torch.sum(e_q_alpha_dns * e_s_gamma.unsqueeze(2), dim=1).unsqueeze(1) * e_s_gamma
    e_proj_norm_dns = torch.norm(e_proj_prime_dns, dim=1, keepdim=True)
    e_proj_dns = e_proj_prime_dns / (e_proj_norm_dns + 1e-8)
    
    # Calculate cos(theta) - angle between e_s_alpha and e_proj
    cos_theta_dns = torch.sum(e_s_alpha * e_proj_dns, dim=1)
    
    # Calculate e_norm for eta angle
    e_norm_dns = torch.cross(e_proj_dns, e_s_gamma, dim=1)
    e_norm_norm_dns = torch.norm(e_norm_dns, dim=1, keepdim=True)
    e_norm_dns = e_norm_dns / (e_norm_norm_dns + 1e-8)
    
    # Calculate cos(eta) - angle between e_q_beta and e_norm
    cos_eta_dns = torch.sum(e_q_beta_dns * e_norm_dns, dim=1)
    
    # Take absolute values as in the paper
    cos_xi_pred_abs = torch.abs(cos_xi_pred)
    cos_theta_pred_abs = torch.abs(cos_theta_pred)
    cos_eta_pred_abs = torch.abs(cos_eta_pred)
    
    cos_xi_dns_abs = torch.abs(cos_xi_dns)
    cos_theta_dns_abs = torch.abs(cos_theta_dns)
    cos_eta_dns_abs = torch.abs(cos_eta_dns)
    
    # Calculate losses according to Eq. 20
    L1_num = torch.sum((cos_xi_dns_abs - cos_xi_pred_abs)**2)
    L1_den = torch.sum(cos_xi_dns_abs**2) + 1e-8
    L1 = L1_num / L1_den
    
    L2_num = torch.sum((cos_theta_dns_abs - cos_theta_pred_abs)**2)
    L2_den = torch.sum(cos_theta_dns_abs**2) + 1e-8
    L2 = L2_num / L2_den
    
    L3_num = torch.sum((cos_eta_dns_abs - cos_eta_pred_abs)**2)
    L3_den = torch.sum(cos_eta_dns_abs**2) + 1e-8
    L3 = L3_num / L3_den
    
    # Total loss
    J = L1 + L2 + L3
    
    return J


def magnitude_loss(psi_pred, psi_dns):
    """
    Calculate magnitude loss (Eq. 22).
    
    Args:
        psi_pred: Predicted magnitude (batch_size, 1)
        psi_dns: Ground truth magnitude (batch_size, 1)
    
    Returns:
        loss: Magnitude loss (scalar)
    """
    # Ensure tensors are properly shaped
    psi_pred = psi_pred.view(-1)
    psi_dns = psi_dns.view(-1)
    
    # Calculate RMSE
    numerator = torch.sum((psi_dns - psi_pred)**2)
    denominator = torch.sum(psi_dns**2) + 1e-8  # Avoid division by zero
    
    return numerator / denominator


def calculate_phi(Q_prime, q_epsilon_sq):
    """
    Calculate phi (Eq. 31) for evaluating model performance.
    
    Args:
        Q_prime: Anisotropic pressure Hessian tensor (batch_size, 3, 3)
        q_epsilon_sq: Second invariant of raw velocity gradient tensor (batch_size)
    
    Returns:
        phi: Conditional average
    """
    # Calculate Q_mn*Q_mn (magnitude)
    Q_magnitude = torch.sum(Q_prime * Q_prime, dim=(1, 2))
    
    # Get standard deviations for normalization
    std_q = torch.std(q_epsilon_sq)
    std_Q_magnitude = torch.std(Q_magnitude)
    
    # Calculate average of Q_magnitude conditioned on q_epsilon_sq
    # This would be done by binning q_epsilon_sq and calculating mean of Q_magnitude in each bin
    # For efficiency, here we just calculate a moving average
    
    # Sort by q_epsilon_sq
    sorted_indices = torch.argsort(q_epsilon_sq)
    sorted_q = q_epsilon_sq[sorted_indices]
    sorted_Q_magnitude = Q_magnitude[sorted_indices]
    
    # Initialize phi
    phi = torch.zeros_like(sorted_q)
    
    # Calculate moving average with fixed window size
    window_size = min(100, len(sorted_q) // 20)  # Adjust window size based on data size
    
    for i in range(len(sorted_q)):
        start = max(0, i - window_size // 2)
        end = min(len(sorted_q), i + window_size // 2)
        phi[i] = torch.mean(sorted_Q_magnitude[start:end])
    
    # Normalize
    phi = phi / std_Q_magnitude * std_q
    
    # Unsort
    inv_sorted_indices = torch.argsort(sorted_indices)
    phi = phi[inv_sorted_indices]
    
    return phi


# --- Realistic Data Generation ---

def generate_reynolds_stress_tensor(num_samples, seed=None, device='cuda'):
    """
    Generate realistic Reynolds stress tensors for incompressible flow.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        device: Device to generate data on
    
    Returns:
        tau: Reynolds stress tensor (batch_size, 3, 3)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random eigenvalues (sorted in descending order)
    # Eigenvalues sum to 1 (normalized) for incompressible flow
    lambda1 = 0.6 + 0.2 * torch.rand(num_samples, device=device)
    lambda3 = 0.1 * torch.rand(num_samples, device=device)
    lambda2 = 1.0 - lambda1 - lambda3
    eigenvalues = torch.stack([lambda1, lambda2, lambda3], dim=1)
    
    # Generate random rotation matrices
    # First, generate random Euler angles
    alpha = 2 * np.pi * torch.rand(num_samples, device=device)
    beta = np.pi * torch.rand(num_samples, device=device)
    gamma = 2 * np.pi * torch.rand(num_samples, device=device)
    
    # Construct rotation matrices
    R = torch.zeros(num_samples, 3, 3, device=device)
    
    # Apply Euler angles to create rotation matrices (ZYZ convention)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    
    # First rotation (around Z)
    R[:, 0, 0] = ca
    R[:, 0, 1] = -sa
    R[:, 1, 0] = sa
    R[:, 1, 1] = ca
    R[:, 2, 2] = 1.0
    
    # Second rotation (around Y)
    R_y = torch.zeros_like(R)
    R_y[:, 0, 0] = cb
    R_y[:, 0, 2] = sb
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = -sb
    R_y[:, 2, 2] = cb
    
    R = torch.bmm(R_y, R)
    
    # Third rotation (around Z)
    R_z = torch.zeros_like(R)
    R_z[:, 0, 0] = cg
    R_z[:, 0, 1] = -sg
    R_z[:, 1, 0] = sg
    R_z[:, 1, 1] = cg
    R_z[:, 2, 2] = 1.0
    
    R = torch.bmm(R_z, R)
    
    # Construct diagonal matrix of eigenvalues
    Lambda = torch.zeros(num_samples, 3, 3, device=device)
    Lambda[:, 0, 0] = eigenvalues[:, 0]
    Lambda[:, 1, 1] = eigenvalues[:, 1]
    Lambda[:, 2, 2] = eigenvalues[:, 2]
    
    # Construct Reynolds stress tensor: tau = R * Lambda * R^T
    tau = torch.bmm(torch.bmm(R, Lambda), R.transpose(1, 2))
    
    return tau


def generate_divergence_free_velocity_gradient(num_samples, seed=None, device='cuda'):
    """
    Generate realistic velocity gradient tensors for incompressible flow (divergence-free).
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        device: Device to generate data on
    
    Returns:
        A: Velocity gradient tensor (batch_size, 3, 3)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random strain-rate tensors (symmetric, trace-free)
    S = torch.randn(num_samples, 3, 3, device=device)
    S = 0.5 * (S + S.transpose(1, 2))  # Make symmetric
    
    # Make trace-free (incompressible flow)
    trace = torch.diagonal(S, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    S = S - trace * torch.eye(3, device=device).unsqueeze(0) / 3.0
    
    # Generate random rotation-rate tensors (anti-symmetric)
    W = torch.randn(num_samples, 3, 3, device=device)
    W = 0.5 * (W - W.transpose(1, 2))  # Make anti-symmetric
    
    # Scale to reasonable magnitudes
    S = S * torch.rand(num_samples, 1, 1, device=device) * 0.1
    W = W * torch.rand(num_samples, 1, 1, device=device) * 0.2
    
    # Velocity gradient tensor A = S + W
    A = S + W
    
    return A


def generate_realistic_training_data(num_samples, seed=None, device='cuda'):
    """
    Generate realistic training data for AIBM model.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        device: Device to generate data on
    
    Returns:
        data: Dictionary of training data tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate velocity gradient tensors
    A_tensor = generate_divergence_free_velocity_gradient(num_samples, seed, device)
    
    # Normalize A tensor
    A_norm, epsilon_sq = normalize_tensor(A_tensor)
    
    # Calculate s_norm and w_norm
    s_norm, w_norm = calculate_s_w(A_norm)
    
    # Calculate tensor invariants
    lambda_invariants = calculate_lambda_invariants(s_norm, w_norm)
    q_r_invariants = calculate_q_r_invariants(s_norm, w_norm)
    
    # Generate realistic pressure Hessian tensor
    # For this example, we'll use a realistic model based on research findings
    # A more accurate approach would involve solving the Poisson equation for pressure
    
    # Generate Q_prime with correct mathematical properties
    Q_prime = torch.zeros_like(A_tensor)
    
    # Step 1: Generate a symmetric tensor
    Q_temp = torch.randn_like(A_tensor)
    Q_temp = 0.5 * (Q_temp + Q_temp.transpose(1, 2))
    
    # Step 2: Make it traceless (anisotropic part is traceless)
    trace = torch.diagonal(Q_temp, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    Q_prime = Q_temp - (trace / 3.0) * torch.eye(3, device=device).unsqueeze(0)
    
    # Step 3: Scale with epsilon_sq to get correct magnitude relationship
    scale_factor = torch.rand(num_samples, 1, 1, device=device) * 0.5 + 0.5
    Q_prime = Q_prime * scale_factor * epsilon_sq.unsqueeze(-1).unsqueeze(-1)
    
    # Step 4: Ensure Q_prime has the correct alignment tendencies with s_norm
    # Here we could implement a more sophisticated alignment algorithm
    # but for simplicity, we'll modify Q_prime to align partially with s_norm
    
    # Get eigenvectors of s_norm
    _, s_eigenvectors = calculate_eigensystem(s_norm)
    
    # Construct aligned Q_prime tensor
    # This is a simplified model; real physics would be more complex
    Q_aligned = torch.zeros_like(Q_prime)
    
    for i in range(num_samples):
        # Get eigenvectors
        e1 = s_eigenvectors[i, :, 0].unsqueeze(1)  # Most extensive (column vector)
        e2 = s_eigenvectors[i, :, 1].unsqueeze(1)  # Intermediate
        e3 = s_eigenvectors[i, :, 2].unsqueeze(1)  # Most compressive
        
        # Construct Q_aligned with tendency to align e_q_gamma with e3 at ~45 degrees
        # and e_q_alpha with e2
        D = torch.diag(torch.tensor([0.5, 1.0, -1.5], device=device))
        
        # Create a slight rotation to misalign from perfect alignment
        angle = torch.tensor(np.pi/4, device=device)  # ~45 degrees
        c, s = torch.cos(angle), torch.sin(angle)
        R = torch.eye(3, device=device)
        R[1, 1] = c
        R[1, 2] = -s
        R[2, 1] = s
        R[2, 2] = c
        
        # Construct eigenvector matrix
        E = torch.cat([e1, e2, e3], dim=1)
        
        # Construct aligned Q_prime: E * R * D * R^T * E^T
        Q_aligned[i] = E @ R @ D @ R.T @ E.T
    
    # Blend original Q_prime with aligned Q_prime
    blend_factor = torch.rand(num_samples, 1, 1, device=device) * 0.3 + 0.7
    Q_prime = blend_factor * Q_aligned + (1 - blend_factor) * Q_prime
    
    # Calculate Q_tilde_prime (normalized Q_prime)
    Q_tilde_prime = normalize_tensor_preserving_shape(Q_prime)
    
    # Calculate psi (magnitude of Q_prime)
    psi = torch.sqrt(torch.sum(Q_prime * Q_prime, dim=(1, 2), keepdim=True))
    
    return {
        'A_tensor': A_tensor,
        'epsilon_sq': epsilon_sq,
        's_norm': s_norm,
        'w_norm': w_norm,
        'lambda_invariants': lambda_invariants,
        'q_r_invariants': q_r_invariants,
        'Q_prime': Q_prime,
        'Q_tilde_prime': Q_tilde_prime,
        'psi': psi
    }


# --- Visualization Utilities ---

def plot_alignment_pdfs(Q_tilde_prime_pred, s_norm, Q_tilde_prime_dns, title_prefix=''):
    """
    Plot PDFs of the cosines of angles between eigenvectors.
    
    Args:
        Q_tilde_prime_pred: Predicted normalized tensor (batch_size, 3, 3)
        s_norm: Normalized strain-rate tensor (batch_size, 3, 3)
        Q_tilde_prime_dns: Ground truth normalized tensor (batch_size, 3, 3)
        title_prefix: Prefix for plot titles
    """
    import matplotlib.pyplot as plt
    
    # Move tensors to CPU for numpy conversion
    Q_tilde_prime_pred = Q_tilde_prime_pred.cpu()
    s_norm = s_norm.cpu()
    Q_tilde_prime_dns = Q_tilde_prime_dns.cpu()
    
    # Calculate eigenvalues and eigenvectors
    _, s_eigenvectors = calculate_eigensystem(s_norm)
    _, Q_pred_eigenvectors = calculate_eigensystem(Q_tilde_prime_pred)
    _, Q_dns_eigenvectors = calculate_eigensystem(Q_tilde_prime_dns)
    
    # Extract eigenvectors
    e_s_alpha = s_eigenvectors[:, :, 0]  # Most extensive
    e_s_beta = s_eigenvectors[:, :, 1]   # Intermediate
    e_s_gamma = s_eigenvectors[:, :, 2]  # Most compressive
    
    e_q_alpha_pred = Q_pred_eigenvectors[:, :, 0]
    e_q_beta_pred = Q_pred_eigenvectors[:, :, 1]
    e_q_gamma_pred = Q_pred_eigenvectors[:, :, 2]
    
    e_q_alpha_dns = Q_dns_eigenvectors[:, :, 0]
    e_q_beta_dns = Q_dns_eigenvectors[:, :, 1]
    e_q_gamma_dns = Q_dns_eigenvectors[:, :, 2]
    
    # Calculate cosines of angles
    # Between s eigenvectors and Q eigenvectors (predicted)
    cos_alpha_alpha_pred = torch.sum(e_s_alpha * e_q_alpha_pred, dim=1).numpy()
    cos_alpha_beta_pred = torch.sum(e_s_alpha * e_q_beta_pred, dim=1).numpy()
    cos_alpha_gamma_pred = torch.sum(e_s_alpha * e_q_gamma_pred, dim=1).numpy()
    
    cos_beta_alpha_pred = torch.sum(e_s_beta * e_q_alpha_pred, dim=1).numpy()
    cos_beta_beta_pred = torch.sum(e_s_beta * e_q_beta_pred, dim=1).numpy()
    cos_beta_gamma_pred = torch.sum(e_s_beta * e_q_gamma_pred, dim=1).numpy()
    
    cos_gamma_alpha_pred = torch.sum(e_s_gamma * e_q_alpha_pred, dim=1).numpy()
    cos_gamma_beta_pred = torch.sum(e_s_gamma * e_q_beta_pred, dim=1).numpy()
    cos_gamma_gamma_pred = torch.sum(e_s_gamma * e_q_gamma_pred, dim=1).numpy()
    
    # Between s eigenvectors and Q eigenvectors (DNS)
    cos_alpha_alpha_dns = torch.sum(e_s_alpha * e_q_alpha_dns, dim=1).numpy()
    cos_alpha_beta_dns = torch.sum(e_s_alpha * e_q_beta_dns, dim=1).numpy()
    cos_alpha_gamma_dns = torch.sum(e_s_alpha * e_q_gamma_dns, dim=1).numpy()
    
    cos_beta_alpha_dns = torch.sum(e_s_beta * e_q_alpha_dns, dim=1).numpy()
    cos_beta_beta_dns = torch.sum(e_s_beta * e_q_beta_dns, dim=1).numpy()
    cos_beta_gamma_dns = torch.sum(e_s_beta * e_q_gamma_dns, dim=1).numpy()
    
    cos_gamma_alpha_dns = torch.sum(e_s_gamma * e_q_alpha_dns, dim=1).numpy()
    cos_gamma_beta_dns = torch.sum(e_s_gamma * e_q_beta_dns, dim=1).numpy()
    cos_gamma_gamma_dns = torch.sum(e_s_gamma * e_q_gamma_dns, dim=1).numpy()
    
    # Create figure
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'{title_prefix}Alignment of Eigenvectors of Q and s', fontsize=16)
    
    # List of all cosines
    cosines_pred = [
        cos_gamma_alpha_pred, cos_gamma_beta_pred, cos_gamma_gamma_pred,
        cos_beta_alpha_pred, cos_beta_beta_pred, cos_beta_gamma_pred,
        cos_alpha_alpha_pred, cos_alpha_beta_pred, cos_alpha_gamma_pred
    ]
    
    cosines_dns = [
        cos_gamma_alpha_dns, cos_gamma_beta_dns, cos_gamma_gamma_dns,
        cos_beta_alpha_dns, cos_beta_beta_dns, cos_beta_gamma_dns,
        cos_alpha_alpha_dns, cos_alpha_beta_dns, cos_alpha_gamma_dns
    ]
    
    titles = [
        r'$e_{c_s} \\cdot e_{\\alpha_p}$', r'$e_{c_s} \\cdot e_{\\beta_p}$', r'$e_{c_s} \\cdot e_{\\gamma_p}$',
        r'$e_{b_s} \\cdot e_{\\alpha_p}$', r'$e_{b_s} \\cdot e_{\\beta_p}$', r'$e_{b_s} \\cdot e_{\\gamma_p}$',
        r'$e_{a_s} \\cdot e_{\\alpha_p}$', r'$e_{a_s} \\cdot e_{\\beta_p}$', r'$e_{a_s} \\cdot e_{\\gamma_p}$'
    ]
    
    # Plot histograms
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            axs[i, j].hist(cosines_dns[idx], bins=30, alpha=0.5, density=True, range=(-1, 1), color='blue', label='DNS')
            axs[i, j].hist(cosines_pred[idx], bins=30, alpha=0.5, density=True, range=(-1, 1), color='red', label='Predicted')
            axs[i, j].set_title(titles[idx])
            axs[i, j].set_xlabel('Cosine of angle')
            axs[i, j].set_ylabel('PDF')
            axs[i, j].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_vorticity_alignment(Q_tilde_prime_pred, w_norm, Q_tilde_prime_dns, title_prefix=''):
    """
    Plot PDFs of the cosines of angles between vorticity and eigenvectors of Q.
    
    Args:
        Q_tilde_prime_pred: Predicted normalized tensor (batch_size, 3, 3)
        w_norm: Normalized rotation-rate tensor (batch_size, 3, 3)
        Q_tilde_prime_dns: Ground truth normalized tensor (batch_size, 3, 3)
        title_prefix: Prefix for plot titles
    """
    import matplotlib.pyplot as plt
    
    # Move tensors to CPU for numpy conversion
    Q_tilde_prime_pred = Q_tilde_prime_pred.cpu()
    w_norm = w_norm.cpu()
    Q_tilde_prime_dns = Q_tilde_prime_dns.cpu()
    
    # Calculate vorticity
    vorticity = calculate_vorticity(w_norm)
    
    # Normalize vorticity
    vorticity_norm = vorticity / (torch.norm(vorticity, dim=1, keepdim=True) + 1e-8)
    
    # Calculate eigenvalues and eigenvectors of Q
    _, Q_pred_eigenvectors = calculate_eigensystem(Q_tilde_prime_pred)
    _, Q_dns_eigenvectors = calculate_eigensystem(Q_tilde_prime_dns)
    
    # Extract eigenvectors
    e_q_alpha_pred = Q_pred_eigenvectors[:, :, 0]
    e_q_beta_pred = Q_pred_eigenvectors[:, :, 1]
    e_q_gamma_pred = Q_pred_eigenvectors[:, :, 2]
    
    e_q_alpha_dns = Q_dns_eigenvectors[:, :, 0]
    e_q_beta_dns = Q_dns_eigenvectors[:, :, 1]
    e_q_gamma_dns = Q_dns_eigenvectors[:, :, 2]
    
    # Calculate cosines of angles
    cos_vort_alpha_pred = torch.sum(vorticity_norm * e_q_alpha_pred, dim=1).numpy()
    cos_vort_beta_pred = torch.sum(vorticity_norm * e_q_beta_pred, dim=1).numpy()
    cos_vort_gamma_pred = torch.sum(vorticity_norm * e_q_gamma_pred, dim=1).numpy()
    
    cos_vort_alpha_dns = torch.sum(vorticity_norm * e_q_alpha_dns, dim=1).numpy()
    cos_vort_beta_dns = torch.sum(vorticity_norm * e_q_beta_dns, dim=1).numpy()
    cos_vort_gamma_dns = torch.sum(vorticity_norm * e_q_gamma_dns, dim=1).numpy()
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{title_prefix}Alignment of Vorticity with Eigenvectors of Q', fontsize=16)
    
    # Plot histograms
    axs[0].hist(cos_vort_alpha_dns, bins=30, alpha=0.5, density=True, range=(-1, 1), color='blue', label='DNS')
    axs[0].hist(cos_vort_alpha_pred, bins=30, alpha=0.5, density=True, range=(-1, 1), color='red', label='Predicted')
    axs[0].set_title(r'$\\omega \\cdot e_{\\alpha_p}$')
    axs[0].set_xlabel('Cosine of angle')
    axs[0].set_ylabel('PDF')
    axs[0].legend()
    
    axs[1].hist(cos_vort_beta_dns, bins=30, alpha=0.5, density=True, range=(-1, 1), color='blue', label='DNS')
    axs[1].hist(cos_vort_beta_pred, bins=30, alpha=0.5, density=True, range=(-1, 1), color='red', label='Predicted')
    axs[1].set_title(r'$\\omega \\cdot e_{\\beta_p}$')
    axs[1].set_xlabel('Cosine of angle')
    axs[1].set_ylabel('PDF')
    axs[1].legend()
    
    axs[2].hist(cos_vort_gamma_dns, bins=30, alpha=0.5, density=True, range=(-1, 1), color='blue', label='DNS')
    axs[2].hist(cos_vort_gamma_pred, bins=30, alpha=0.5, density=True, range=(-1, 1), color='red', label='Predicted')
    axs[2].set_title(r'$\\omega \\cdot e_{\\gamma_p}$')
    axs[2].set_xlabel('Cosine of angle')
    axs[2].set_ylabel('PDF')
    axs[2].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_psi_pdf(psi_pred, psi_dns):
    """
    Plot PDF of the magnitude (psi) of the pressure Hessian tensor.
    
    Args:
        psi_pred: Predicted magnitude (batch_size, 1)
        psi_dns: Ground truth magnitude (batch_size, 1)
    """
    import matplotlib.pyplot as plt
    
    # Move tensors to CPU for numpy conversion
    psi_pred = psi_pred.cpu().numpy().flatten()
    psi_dns = psi_dns.cpu().numpy().flatten()
    
    # Determine max value for x-axis
    max_val = max(np.percentile(psi_dns, 99), np.percentile(psi_pred, 99))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(psi_dns, bins=50, alpha=0.5, density=True, range=(0, max_val), color='blue', label='DNS')
    ax.hist(psi_pred, bins=50, alpha=0.5, density=True, range=(0, max_val), color='red', label='Predicted')
    
    ax.set_title('PDF of ψ (Magnitude of Anisotropic Pressure Hessian Tensor)')
    ax.set_xlabel('ψ')
    ax.set_ylabel('PDF')
    ax.legend()
    
    return fig


def plot_phi_vs_q(phi_pred, phi_dns, q_epsilon_sq):
    """
    Plot phi vs q*epsilon^2 for model evaluation. (Fig. 12 in paper)
    
    Args:
        phi_pred: Predicted phi (batch_size)
        phi_dns: Ground truth phi (batch_size)
        q_epsilon_sq: Second invariant of raw velocity gradient tensor (batch_size)
    """
    import matplotlib.pyplot as plt
    
    # Move tensors to CPU for numpy conversion
    phi_pred = phi_pred.cpu().numpy()
    phi_dns = phi_dns.cpu().numpy()
    q_epsilon_sq = q_epsilon_sq.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by q_epsilon_sq
    sorted_indices = np.argsort(q_epsilon_sq)
    sorted_q = q_epsilon_sq[sorted_indices]
    sorted_phi_dns = phi_dns[sorted_indices]
    sorted_phi_pred = phi_pred[sorted_indices]
    
    # Plot
    ax.scatter(sorted_q, sorted_phi_dns, s=10, color='blue', label='DNS')
    ax.scatter(sorted_q, sorted_phi_pred, s=10, color='red', label='AIBM')
    
    ax.set_title('Plot of φ vs q·ε²')
    ax.set_xlabel('q·ε²')
    ax.set_ylabel('φ')
    ax.legend()
    
    # Set x-axis limits to focus on the region of interest
    ax.set_xlim(-1.0, 1.0)
    
    return fig


# --- Main Script ---

def main():
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set parameters
    config = {
        # TBNN Alignment Network
        'tbnn_input_dim': 5,
        'tbnn_hidden_layers': [50, 100, 100, 100, 50],
        'tbnn_output_coeffs': 10,
        'tbnn_batch_size': 16384,
        'tbnn_lr': 1e-3,
        'tbnn_dropout': 0.1,
        
        # Magnitude Network
        'mag_input_dim': 2,
        'mag_hidden_layers': [50, 80, 50],
        'mag_batch_size': 4096,
        'mag_lr': 5e-3,
        'mag_dropout': 0.1,
        
        # Training
        'epochs': 10,  # Reduced for quick demo (paper uses 400)
        'early_stopping_patience': 5,
    }
    
    # Generate training data
    print("Generating training data...")
    num_train_samples = 250000  # Paper uses 250000
    train_data = generate_realistic_training_data(num_train_samples, seed=42, device=device)
    
    # Generate validation data
    print("Generating validation data...")
    num_val_samples = 50000  # Paper uses 50000
    val_data = generate_realistic_training_data(num_val_samples, seed=43, device=device)
    
    # Initialize AIBM model
    print("Initializing AIBM model...")
    aibm = AugmentedInvariantBasedModel(device=device, config=config)
    
    # Train model
    print("Training AIBM model...")
    aibm.train(train_data, val_data)
    
    # Generate test data
    print("Generating test data...")
    num_test_samples = 100000
    test_data = generate_realistic_training_data(num_test_samples, seed=44, device=device)
    
    # Evaluate model
    print("Evaluating AIBM model...")
    eval_results = aibm.evaluate(test_data)
    
    print(f"Alignment Loss: {eval_results['alignment_loss']:.6f}")
    print(f"Magnitude Loss: {eval_results['magnitude_loss']:.6f}")
    print(f"RMSE: {eval_results['rmse']:.6f}")
    
    # Plot results
    print("Plotting results...")
    
    # Extract tensors for plotting
    Q_tilde_prime_pred = eval_results['Q_tilde_prime_pred']
    psi_pred = eval_results['psi_pred']
    Q_prime_pred = eval_results['Q_prime_pred']
    
    Q_tilde_prime_dns = test_data['Q_tilde_prime'].cpu()
    psi_dns = test_data['psi'].cpu()
    Q_prime_dns = test_data['Q_prime'].cpu()
    s_norm = test_data['s_norm'].cpu()
    w_norm = test_data['w_norm'].cpu()
    
    # Calculate q_epsilon_sq for phi plots
    q_r_invariants = test_data['q_r_invariants'].cpu()
    q = q_r_invariants[:, 0]
    epsilon_sq = test_data['epsilon_sq'].cpu()
    q_epsilon_sq = q * epsilon_sq
    
    # Calculate phi for DNS and predicted Q_prime
    phi_dns = calculate_phi(Q_prime_dns, q_epsilon_sq)
    phi_pred = calculate_phi(Q_prime_pred, q_epsilon_sq)
    
    # Plot alignment PDFs
    fig_alignment = plot_alignment_pdfs(Q_tilde_prime_pred[:10000], s_norm[:10000], Q_tilde_prime_dns[:10000])
    fig_alignment.savefig('alignment_pdfs.png')
    
    # Plot vorticity alignment
    fig_vorticity = plot_vorticity_alignment(Q_tilde_prime_pred[:10000], w_norm[:10000], Q_tilde_prime_dns[:10000])
    fig_vorticity.savefig('vorticity_alignment.png')
    
    # Plot psi PDF
    fig_psi = plot_psi_pdf(psi_pred[:10000], psi_dns[:10000])
    fig_psi.savefig('psi_pdf.png')
    
    # Plot phi vs q
    fig_phi = plot_phi_vs_q(phi_pred[:10000], phi_dns[:10000], q_epsilon_sq[:10000])
    fig_phi.savefig('phi_vs_q.png')
    
    print("Done!")


if __name__ == "__main__":
    main()
