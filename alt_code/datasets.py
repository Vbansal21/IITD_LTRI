# datasets.py

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io

class MatlabTurbulenceDataset(Dataset):
    """
    PyTorch Dataset for loading turbulence data from .mat files.
    Loads velocity gradient (A) and pressure Hessian (PH), computes anisotropic Q.
    Supports optional random noising/masking/corruption for robust training and distillation.

    Args:
        vel_grad_path (str): Path to 'velGrad.mat' file (e.g., (9, N) or (N, 9))
        pressure_hessian_path (str): Path to 'PH.mat' file (e.g., (N, 9))
        device (torch.device): Device for loading tensors
        noising: Dict for noise type and params, e.g. {'type': 'gaussian', 'std': 0.05}
        masking: Dict for masking/corruption, e.g. {'type': 'block', 'prob': 0.2}
        shuffle_on_init (bool): Shuffle samples at initialization
        sample_transform: Optional callable applied to (A, Q) after loading but before __getitem__
    """
    def __init__(self, vel_grad_path, pressure_hessian_path, device=None,
                 noising=None, masking=None, shuffle_on_init=False, sample_transform=None):
        super().__init__()
        print(f"Loading velocity gradient from: {vel_grad_path}")
        print(f"Loading pressure Hessian from: {pressure_hessian_path}")

        # --- Load .mat files ---
        vel_grad_mat = scipy.io.loadmat(vel_grad_path)
        ph_mat = scipy.io.loadmat(pressure_hessian_path)

        # Try common key guesses; modify as needed for your dataset!
        key_A = [k for k in vel_grad_mat.keys() if not k.startswith('__')][0]
        key_PH = [k for k in ph_mat.keys() if not k.startswith('__')][0]
        vel_grad = vel_grad_mat[key_A] # (9,N) or (N,9) or (N,3,3)
        ph = ph_mat[key_PH]

        # --- Format to (N,3,3) ---
        def to_3x3(data):
            data = np.asarray(data)
            if data.ndim == 2 and data.shape[0] == 9:
                data = data.T # (N,9)
            if data.ndim == 2 and data.shape[1] == 9:
                data = data.reshape(-1,3,3)
            elif data.ndim == 3 and data.shape[1:] == (3,3):
                pass # already fine
            else:
                raise ValueError(f"Unrecognized shape for data: {data.shape}")
            return data

        A = to_3x3(vel_grad).astype(np.float32)   # (N,3,3)
        PH = to_3x3(ph).astype(np.float32)        # (N,3,3)
        N = A.shape[0]
        assert PH.shape[0] == N, "Mismatch between A and PH samples!"

        # --- Convert PH to anisotropic Q (traceless) ---
        trace_PH = np.trace(PH, axis1=1, axis2=2) # (N,)
        eye = np.eye(3, dtype=np.float32)
        Q = PH - trace_PH[:, None, None] / 3. * eye[None, :, :]

        # --- Save raw tensors ---
        self.A_raw = torch.from_numpy(A)      # (N,3,3)
        self.Q_raw = torch.from_numpy(Q)      # (N,3,3)
        self.N = N
        self.device = device if device is not None else torch.device('cpu')

        # --- Optional full-data transform ---
        if sample_transform:
            self.A_raw, self.Q_raw = sample_transform(self.A_raw, self.Q_raw)

        # --- Optional: Shuffle dataset order at load ---
        idx = np.arange(self.N)
        if shuffle_on_init:
            np.random.shuffle(idx)
            self.A_raw = self.A_raw[idx]
            self.Q_raw = self.Q_raw[idx]

        # --- Noising/Masking settings ---
        self.noising = noising
        self.masking = masking

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        A = self.A_raw[idx].clone().to(self.device)
        Q = self.Q_raw[idx].clone().to(self.device)

        # --- Optional on-the-fly noising ---
        if self.noising is not None:
            noise_type = self.noising.get('type', None)
            if noise_type == 'gaussian':
                std = self.noising.get('std', 0.01)
                A += torch.randn_like(A) * std
                Q += torch.randn_like(Q) * std
            elif noise_type == 'uniform':
                eps = self.noising.get('eps', 0.01)
                A += (torch.rand_like(A)-0.5)*2*eps
                Q += (torch.rand_like(Q)-0.5)*2*eps
            # Add other noise types as needed

        # --- Optional masking/corruption ---
        if self.masking is not None:
            mask_type = self.masking.get('type', None)
            if mask_type == 'block':
                prob = self.masking.get('prob', 0.1)
                if torch.rand(1).item() < prob:
                    # zero-out a random sub-block (row/column)
                    row = torch.randint(0,3,(1,)).item()
                    col = torch.randint(0,3,(1,)).item()
                    A[row,:] = 0
                    A[:,col] = 0
                    Q[row,:] = 0
                    Q[:,col] = 0
            elif mask_type == 'random':
                p = self.masking.get('prob', 0.1)
                mask = (torch.rand_like(A) > p).float()
                A = A * mask
                Q = Q * mask
            # Add other masking types as needed

        return A, Q

    def get_all(self):
        """Return all A, Q in shape (N,3,3), useful for full-dataset evaluation."""
        return self.A_raw.clone().to(self.device), self.Q_raw.clone().to(self.device)

    def as_tensor_dataset(self):
        """Return the full set as a torch.utils.data.TensorDataset (for validation/test)."""
        return torch.utils.data.TensorDataset(
            self.A_raw.clone().to(self.device), self.Q_raw.clone().to(self.device)
        )

# --- Optional: Fast in-memory "dummy" dataset for debug ---
class DummyTurbulenceDataset(Dataset):
    """
    Synthetic turbulence data for debugging pipelines. Generates random traceless A, Q.
    Args:
        N: number of samples
        device: cpu/cuda
    """
    def __init__(self, N=10000, device=None, seed=42):
        super().__init__()
        device = device if device is not None else torch.device('cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Random tensors, then traceless
        A = torch.randn(N,3,3)
        Q = torch.randn(N,3,3)
        eye = torch.eye(3)
        A -= torch.einsum('bii->b', A)[:,None,None] * eye / 3
        Q -= torch.einsum('bii->b', Q)[:,None,None] * eye / 3
        self.A_raw = A.to(device)
        self.Q_raw = Q.to(device)
        self.N = N
        self.device = device

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.A_raw[idx].clone(), self.Q_raw[idx].clone()

    def get_all(self):
        return self.A_raw.clone(), self.Q_raw.clone()

# --- End of file ---
