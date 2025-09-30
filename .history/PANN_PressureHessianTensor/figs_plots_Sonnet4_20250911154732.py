import numpy as np
import scipy.io
import matplotlib.pyplot as plt

class MatlabDataset:
    def __init__(self, vel_grad_path, pressure_hessian_path):
        print(f"Loading data from {vel_grad_path} and {pressure_hessian_path}...")
        
        def load_mat_data(path):
            mat = scipy.io.loadmat(path)
            key = next(k for k in mat if k not in ('__header__', '__version__', '__globals__'))
            return mat[key].astype(np.float32)
        
        vel_grad_data = load_mat_data(vel_grad_path)
        if vel_grad_data.shape[0] == 9: 
            vel_grad_data = vel_grad_data.T
        self.A = vel_grad_data.reshape(-1, 3, 3)
        
        ph_data = load_mat_data(pressure_hessian_path)
        if ph_data.shape[0] == 9: 
            ph_data = ph_data.T
        raw_P = ph_data.reshape(-1, 3, 3)
        
        # Remove trace to get anisotropic part Q
        trace_P = np.einsum('bii->b', raw_P)[:, None, None]
        self.Q = raw_P - (trace_P / 3.0) * np.eye(3)[None, :, :]
        
        assert self.A.shape[0] == self.Q.shape[0], "Data sample counts do not match."
        self.num_samples = self.A.shape[0]
        print(f"Data loaded successfully. Found {self.num_samples} samples.")
    
    def __len__(self): 
        return self.num_samples
    
    def __getitem__(self, idx): 
        return self.A[idx], self.Q[idx]

def safe_eigen_decomposition(tensors):
    """Compute eigenvalues and eigenvectors with numerical stability"""
    eigenvalues = []
    eigenvectors = []
    
    for tensor in tensors:
        # Ensure tensor is symmetric for strain rate
        if not np.allclose(tensor, tensor.T, atol=1e-10):
            tensor = 0.5 * (tensor + tensor.T)
        
        try:
            vals, vecs = np.linalg.eigh(tensor)
            # Sort in descending order (largest to smallest eigenvalue)
            idx = np.argsort(vals)[::-1]
            vals = vals[idx]
            vecs = vecs[:, idx]
            eigenvalues.append(vals)
            eigenvectors.append(vecs)
        except np.linalg.LinAlgError:
            # Fallback for problematic cases
            vals = np.array([0.0, 0.0, 0.0])
            vecs = np.eye(3)
            eigenvalues.append(vals)
            eigenvectors.append(vecs)
    
    return np.array(eigenvalues), np.array(eigenvectors)

def compute_direction_cosine(v1, v2):
    """Compute absolute cosine of angle between unit vectors"""
    # Normalize vectors to handle numerical issues
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12)
    
    cos_theta = np.abs(np.einsum('ik,ik->i', v1_norm, v2_norm))
    # Clamp to [0,1] to handle numerical errors
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    return cos_theta

def get_eigenvector_alignments(s_tensor, Q_hat_prime):
    """Compute alignments between strain rate and pressure Hessian eigenvectors (Figure 9)"""
    
    evals_s, evecs_s = safe_eigen_decomposition(s_tensor)
    evals_Q, evecs_Q = safe_eigen_decomposition(Q_hat_prime)
    
    # Paper notation: a_s > b_s > c_s and a_p > b_p > c_p (descending order)
    e_as, e_bs, e_cs = evecs_s[:, :, 0], evecs_s[:, :, 1], evecs_s[:, :, 2]
    e_ap, e_bp, e_cp = evecs_Q[:, :, 0], evecs_Q[:, :, 1], evecs_Q[:, :, 2]
    
    alignments = {}
    # Figure 9 alignment combinations
    alignments['ecs_eap'] = compute_direction_cosine(e_cs, e_ap)
    alignments['ecs_ebp'] = compute_direction_cosine(e_cs, e_bp)
    alignments['ecs_ecp'] = compute_direction_cosine(e_cs, e_cp)
    
    alignments['ebs_eap'] = compute_direction_cosine(e_bs, e_ap)
    alignments['ebs_ebp'] = compute_direction_cosine(e_bs, e_bp)
    alignments['ebs_ecp'] = compute_direction_cosine(e_bs, e_cp)
    
    alignments['eas_eap'] = compute_direction_cosine(e_as, e_ap)
    alignments['eas_ebp'] = compute_direction_cosine(e_as, e_bp)
    alignments['eas_ecp'] = compute_direction_cosine(e_as, e_cp)
    
    return alignments

def get_vorticity_vector(w_tensor):
    """Extract normalized vorticity vector from rotation rate tensor"""
    # Vorticity components: ω_i = ε_ijk w_jk / 2
    omega_x = w_tensor[:, 2, 1] - w_tensor[:, 1, 2]
    omega_y = w_tensor[:, 0, 2] - w_tensor[:, 2, 0] 
    omega_z = w_tensor[:, 1, 0] - w_tensor[:, 0, 1]
    
    omega = np.stack([omega_x, omega_y, omega_z], axis=1)
    omega_norm = np.linalg.norm(omega, axis=1, keepdims=True)
    
    # Normalize with small epsilon to prevent division by zero
    return omega / (omega_norm + 1e-12)

def get_vorticity_alignments(Q_hat_prime, w_tensor):
    """Compute alignments between vorticity vector and pressure Hessian eigenvectors (Figure 10)"""
    
    omega = get_vorticity_vector(w_tensor)
    _, evecs_Q = safe_eigen_decomposition(Q_hat_prime)
    
    e_ap, e_bp, e_cp = evecs_Q[:, :, 0], evecs_Q[:, :, 1], evecs_Q[:, :, 2]
    
    alignments = {}
    alignments['vorticity_eap'] = compute_direction_cosine(omega, e_ap)
    alignments['vorticity_ebp'] = compute_direction_cosine(omega, e_bp)
    alignments['vorticity_ecp'] = compute_direction_cosine(omega, e_cp)
    
    return alignments

def compute_w_statistic(Q_tensor, A_tensor):
    """Compute w = sqrt(Q'_ij Q'_ij) where Q' = Q/e² (Figure 11)"""
    
    # Compute e = sqrt(A_ij A_ij)
    e_mag = np.sqrt(np.sum(A_tensor * A_tensor, axis=(1,2)))
    
    # Avoid division by zero
    e_mag = np.maximum(e_mag, 1e-12)
    
    # Q' = Q / e²
    Q_prime = Q_tensor / (e_mag[:, None, None] ** 2)
    
    # w = sqrt(Q'_ij Q'_ij)
    w = np.sqrt(np.sum(Q_prime * Q_prime, axis=(1,2)))
    
    return w

def compute_phi_statistic(Q_tensor, A_tensor, n_bins=50):
    """Compute φ vs qe² statistic (Figure 12)"""
    
    # Compute e = sqrt(A_ij A_ij)
    e_mag = np.sqrt(np.sum(A_tensor * A_tensor, axis=(1,2)))
    e_mag = np.maximum(e_mag, 1e-12)
    
    # Compute normalized velocity gradient a = A/e
    a_tensor = A_tensor / e_mag[:, None, None]
    
    # Compute second invariant q = 0.5*(p² - Tr(a²)) where p = Tr(a) = 0 for incompressible
    p = np.trace(a_tensor, axis1=1, axis2=2)
    Tr_a2 = np.einsum('bij,bji->b', a_tensor, a_tensor)
    q = 0.5 * (p*p - Tr_a2)
    
    # Compute qe²
    qe2 = q * (e_mag ** 2)
    
    # Compute Q_mn Q_mn
    Q_frobenius_sq = np.sum(Q_tensor * Q_tensor, axis=(1,2))
    
    # Create bins and compute conditional averages
    qe2_min, qe2_max = np.percentile(qe2, [1, 99])  # Use percentiles to avoid outliers
    bins = np.linspace(qe2_min, qe2_max, n_bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Digitize data
    bin_indices = np.digitize(qe2, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
    
    # Compute conditional averages
    phi_values = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        mask = bin_indices == i
        if np.sum(mask) > 10:  # Require minimum samples per bin
            phi_values[i] = np.mean(Q_frobenius_sq[mask])
        else:
            phi_values[i] = np.nan
    
    # Remove NaN values
    valid_mask = ~np.isnan(phi_values)
    return bin_centers[valid_mask], phi_values[valid_mask]

def plot_figure_9(alignments):
    """Plot Figure 9: Eigenvector alignments between strain rate and pressure Hessian"""
    
    labels = [
        ('ecs_eap', r'$\mathbf{\hat{e}_c^s} \cdot \mathbf{\hat{e}_a^p}$'),
        ('ecs_ebp', r'$\mathbf{\hat{e}_c^s} \cdot \mathbf{\hat{e}_b^p}$'), 
        ('ecs_ecp', r'$\mathbf{\hat{e}_c^s} \cdot \mathbf{\hat{e}_c^p}$'),
        ('ebs_eap', r'$\mathbf{\hat{e}_b^s} \cdot \mathbf{\hat{e}_a^p}$'),
        ('ebs_ebp', r'$\mathbf{\hat{e}_b^s} \cdot \mathbf{\hat{e}_b^p}$'),
        ('ebs_ecp', r'$\mathbf{\hat{e}_b^s} \cdot \mathbf{\hat{e}_c^p}$'),
        ('eas_eap', r'$\mathbf{\hat{e}_a^s} \cdot \mathbf{\hat{e}_a^p}$'),
        ('eas_ebp', r'$\mathbf{\hat{e}_a^s} \cdot \mathbf{\hat{e}_b^p}$'),
        ('eas_ecp', r'$\mathbf{\hat{e}_a^s} \cdot \mathbf{\hat{e}_c^p}$')
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    bins = np.linspace(0, 1, 50)
    
    for i, (key, title) in enumerate(labels):
        ax = axes[i//3, i%3]
        data = alignments[key]
        
        # Remove any potential NaN values
        data_clean = data[~np.isnan(data)]
        
        ax.hist(data_clean, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('|cos θ|')
        ax.set_ylabel('PDF')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 9: Eigenvector Alignments (Strain Rate vs Pressure Hessian)', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_figure_10(alignments):
    """Plot Figure 10: Vorticity alignments with pressure Hessian eigenvectors"""
    
    labels = [
        ('vorticity_eap', r'$\boldsymbol{\omega} \cdot \mathbf{\hat{e}_a^p}$'),
        ('vorticity_ebp', r'$\boldsymbol{\omega} \cdot \mathbf{\hat{e}_b^p}$'),
        ('vorticity_ecp', r'$\boldsymbol{\omega} \cdot \mathbf{\hat{e}_c^p}$')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(0, 1, 50)
    
    for i, (key, title) in enumerate(labels):
        ax = axes[i]
        data = alignments[key]
        
        data_clean = data[~np.isnan(data)]
        
        ax.hist(data_clean, bins=bins, density=True, alpha=0.7, color='red', edgecolor='black')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('|cos θ|')
        ax.set_ylabel('PDF')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 10: Vorticity Vector Alignments with Pressure Hessian Eigenvectors', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_figure_11(w_values):
    """Plot Figure 11: PDF of w statistic"""
    
    # Remove outliers and NaN values
    w_clean = w_values[~np.isnan(w_values)]
    w_clean = w_clean[w_clean < np.percentile(w_clean, 99)]  # Remove top 1% outliers
    
    bins = np.linspace(0, np.max(w_clean), 50)
    
    plt.figure(figsize=(8, 6))
    plt.hist(w_clean, bins=bins, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title('Figure 11: PDF of w statistic', fontsize=14)
    plt.xlabel('w')
    plt.ylabel('PDF')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_figure_12(bins, phi_values):
    """Plot Figure 12: φ vs qe² statistic"""
    
    plt.figure(figsize=(8, 6))
    plt.plot(bins, phi_values, 'o-', color='purple', markersize=4, linewidth=2)
    plt.title('Figure 12: φ statistic vs qe²', fontsize=14)
    plt.xlabel('qe²')
    plt.ylabel('φ')
    plt.grid(True, alpha=0.3)
    plt.show()

def main_analysis(vel_grad_path='velGrad.mat', pressure_hessian_path='PH.mat'):
    """Main function to load data and generate all plots"""
    
    print("Starting analysis...")
    
    # Load dataset
    dataset = MatlabDataset(vel_grad_path, pressure_hessian_path)
    A_data, Q_data = dataset.A, dataset.Q
    
    print(f"Loaded {len(A_data)} samples")
    
    # Compute strain rate (s) and rotation rate (w) tensors
    s_tensor = 0.5 * (A_data + np.transpose(A_data, (0, 2, 1)))
    w_tensor = 0.5 * (A_data - np.transpose(A_data, (0, 2, 1)))
    
    # Compute e magnitude and Q' 
    e_mag = np.sqrt(np.sum(A_data * A_data, axis=(1,2)))
    e_mag = np.maximum(e_mag, 1e-12)
    Q_prime = Q_data / (e_mag[:, None, None] ** 2)
    
    # Compute Q̂' (self-normalized)
    Q_prime_mag = np.sqrt(np.sum(Q_prime * Q_prime, axis=(1,2)))
    Q_prime_mag = np.maximum(Q_prime_mag, 1e-12)
    Q_hat_prime = Q_prime / Q_prime_mag[:, None, None]
    
    print("Computing alignments for Figure 9...")
    alignments_fig9 = get_eigenvector_alignments(s_tensor, Q_hat_prime)
    
    print("Computing alignments for Figure 10...")
    alignments_fig10 = get_vorticity_alignments(Q_hat_prime, w_tensor)
    
    print("Computing w statistic for Figure 11...")
    w_values = compute_w_statistic(Q_data, A_data)
    
    print("Computing φ statistic for Figure 12...")
    bins, phi_values = compute_phi_statistic(Q_data, A_data)
    
    # Generate plots
    print("Generating Figure 9...")
    plot_figure_9(alignments_fig9)
    
    print("Generating Figure 10...")
    plot_figure_10(alignments_fig10)
    
    print("Generating Figure 11...")
    plot_figure_11(w_values)
    
    print("Generating Figure 12...")
    plot_figure_12(bins, phi_values)
    
    print("Analysis complete!")
    
    return alignments_fig9, alignments_fig10, w_values, bins, phi_values

# Usage example:
# results = main_analysis('velGrad.mat', 'PH.mat')
