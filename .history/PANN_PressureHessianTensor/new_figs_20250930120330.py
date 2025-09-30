#!/usr/bin/env python3
import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

EPS = 1e-14

def load_mat(path):
    """Load .mat file, extract first non-metadata key"""
    mat = scipy.io.loadmat(path)
    key = [k for k in mat if not k.startswith('__')][0]
    data = np.asarray(mat[key], dtype=np.float64)
    # Handle both (9,N) and (N,9)
    if data.shape[0] == 9:
        data = data.T
    assert data.shape[1] == 9, f"Expected Nx9, got {data.shape}"
    return data

def reshape_tensors(V, mapping='matlab_col'):
    """
    V: (N, 9) flattened tensors
    mapping: 'matlab_col' = M(:) column-major
             'row_major' = row-major flattening
    Returns: (N, 3, 3)
    """
    N = V.shape[0]
    T = np.zeros((N, 3, 3), dtype=np.float64)
    
    if mapping == 'matlab_col':
        # MATLAB M(:) stacks columns: [M11 M21 M31 M12 M22 M32 M13 M23 M33]
        T[:, :, 0] = V[:, 0:3]
        T[:, :, 1] = V[:, 3:6]
        T[:, :, 2] = V[:, 6:9]
    else:  # row_major
        T[:, 0, :] = V[:, 0:3]
        T[:, 1, :] = V[:, 3:6]
        T[:, 2, :] = V[:, 6:9]
    
    return T

def detect_mapping(V_A, V_Q, sample_size=50000):
    """Auto-detect flattening convention via incompressibility"""
    N = min(V_A.shape[0], sample_size)
    idx = np.random.choice(V_A.shape[0], N, replace=False)
    
    scores = {}
    for mapping in ['matlab_col', 'row_major']:
        A = reshape_tensors(V_A[idx], mapping)
        Q = reshape_tensors(V_Q[idx], mapping)
        
        # Incompressibility: Tr[A] ≈ 0
        tr_A = np.abs(A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2])
        tr_Q = np.abs(Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2])
        
        # Q should be trace-free, A should be divergence-free
        score = np.median(tr_A) + np.median(tr_Q)
        scores[mapping] = score
        
        print(f"{mapping}: median|Tr[A]|={np.median(tr_A):.2e}, median|Tr[Q]|={np.median(tr_Q):.2e}")
    
    best = min(scores, key=scores.get)
    print(f"→ Selected: {best}\n")
    return best

# ============ Physics ============

def compute_normalized_tensors(A_np, Q_np):
    """
    Compute all normalized quantities from paper:
    - e² = A_ij A_ij
    - a = A/e (normalized vel-grad)
    - s = sym(a), w = antisym(a)
    - Q⁰ = Q/e² (normalized aniso PH)
    - ω = ||Q⁰||_F
    - Q̂⁰ = Q⁰/ω (unit-norm orientation)
    """
    A = torch.from_numpy(A_np).to(torch.float64)
    Q = torch.from_numpy(Q_np).to(torch.float64)
    
    # Magnitude
    e2 = torch.sum(A * A, dim=(-2, -1))  # (N,)
    e = torch.sqrt(e2.clamp(min=EPS))
    
    # Normalized velocity gradient
    a = A / e.view(-1, 1, 1)
    s = 0.5 * (a + a.transpose(-1, -2))
    w = 0.5 * (a - a.transpose(-1, -2))
    
    # Normalized anisotropic pressure Hessian
    Q0 = Q / e2.view(-1, 1, 1)
    omega = torch.sqrt(torch.sum(Q0 * Q0, dim=(-2, -1)).clamp(min=EPS))
    Qhat0 = Q0 / omega.view(-1, 1, 1)
    
    return {
        'a': a, 's': s, 'w': w,
        'Q0': Q0, 'Qhat0': Qhat0, 'omega': omega,
        'e2': e2
    }

def compute_eigensystem(M_torch):
    """
    Compute eigenvalues/eigenvectors sorted descending.
    Returns (N,3) eigenvalues, (N,3,3) eigenvectors-as-columns
    """
    evals, evecs = torch.linalg.eigh(M_torch)  # ascending
    idx = torch.argsort(evals, dim=-1, descending=True)
    
    evals_sorted = torch.gather(evals, -1, idx)
    idx_exp = idx.unsqueeze(-2).expand(-1, 3, -1)
    evecs_sorted = torch.gather(evecs, -1, idx_exp)
    
    return evals_sorted, evecs_sorted

def compute_vorticity(w_torch):
    """
    From antisymmetric w = (a - a^T)/2:
    ω_vec = [w_32, w_13, w_21]  (components of curl)
    """
    vort = torch.stack([
        w_torch[:, 2, 1],  # ω_x
        w_torch[:, 0, 2],  # ω_y
        w_torch[:, 1, 0],  # ω_z
    ], dim=1)
    return torch.nn.functional.normalize(vort, dim=-1, eps=EPS)

# ============ Figures ============

def fig9_alignment_s_Qhat0(A_np, Q_np, bins=80):
    """
    Fig. 9: PDF of |cos θ| between eigenvectors of s and Q̂⁰
    9 subplots: 3 eigenvectors of s × 3 eigenvectors of Q̂⁰
    """
    tens = compute_normalized_tensors(A_np, Q_np)
    _, E_s = compute_eigensystem(tens['s'])
    _, E_Q = compute_eigensystem(tens['Qhat0'])
    
    # Normalize eigenvectors
    E_s = torch.nn.functional.normalize(E_s, dim=-2, eps=EPS)
    E_Q = torch.nn.functional.normalize(E_Q, dim=-2, eps=EPS)
    
    # Dot products: (N, 3, 3) where [i,j] = |E_s[:,i] · E_Q[:,j]|
    cos_mat = torch.abs(torch.einsum('nai,naj->nij', E_s, E_Q)).cpu().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    labels_s = ['$\\hat{e}_a^s$ (max)', '$\\hat{e}_b^s$ (mid)', '$\\hat{e}_c^s$ (min)']
    labels_Q = ['$\\hat{e}_a^p$ (max)', '$\\hat{e}_b^p$ (mid)', '$\\hat{e}_c^p$ (min)']
    
    for i in range(3):
        for j in range(3):
            cos_vals = cos_mat[:, i, j]
            ax = axes[i, j]
            ax.hist(cos_vals, bins=bins, range=(0, 1), density=True, 
                   histtype='step', lw=1.5, color='C0')
            ax.set_xlim(0, 1)
            ax.set_xlabel('$|\\cos\\theta|$')
            ax.set_ylabel('PDF')
            ax.set_title(f'{labels_s[i]} vs {labels_Q[j]}')
            ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def fig10_alignment_vorticity_Qhat0(A_np, Q_np, bins=80):
    """
    Fig. 10: PDF of |cos θ| between vorticity vector and eigenvectors of Q̂⁰
    3 subplots: one per eigenvector of Q̂⁰
    """
    tens = compute_normalized_tensors(A_np, Q_np)
    vort = compute_vorticity(tens['w'])  # (N, 3)
    _, E_Q = compute_eigensystem(tens['Qhat0'])  # (N, 3, 3)
    
    E_Q = torch.nn.functional.normalize(E_Q, dim=-2, eps=EPS)
    
    # |vort · E_Q[:,:,j]| for j=0,1,2
    cos_vals = torch.abs(torch.einsum('ni,nij->nj', vort, E_Q)).cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ['$\\hat{e}_a^p$ (max)', '$\\hat{e}_b^p$ (mid)', '$\\hat{e}_c^p$ (min)']
    
    for j, ax in enumerate(axes):
        ax.hist(cos_vals[:, j], bins=bins, range=(0, 1), density=True,
               histtype='step', lw=1.5, color='C0')
        ax.set_xlim(0, 1)
        ax.set_xlabel('$|\\cos\\theta|$')
        ax.set_ylabel('PDF')
        ax.set_title(f'$\\hat{{\\omega}}$ vs {labels[j]}')
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def fig11_pdf_omega(A_np, Q_np, bins=100, clip_percentile=99.5):
    """
    Fig. 11: PDF of ω = ||Q⁰||_F
    """
    tens = compute_normalized_tensors(A_np, Q_np)
    omega = tens['omega'].cpu().numpy()
    
    # Clip outliers
    omega_max = np.percentile(omega, clip_percentile)
    omega_clipped = omega[omega <= omega_max]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(omega_clipped, bins=bins, density=True, histtype='step', lw=1.5, color='C0')
    ax.set_xlabel('$\\omega = ||Q^0||_F$')
    ax.set_ylabel('PDF')
    ax.set_title('Fig. 11: PDF of $\\omega$')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def fig12_phi_vs_qe2(A_np, Q_np, bins=60, clip_percentile=2.0):
    """
    Fig. 12: φ(qe²) vs qe²
    where φ = E[Q_ij Q_ij | qe²] / (σ_qe² × σ_{Q_ij Q_ij})
    and qe² = -0.5 Tr[A²] (second invariant of raw A)
    """
    A = torch.from_numpy(A_np).to(torch.float64)
    Q = torch.from_numpy(Q_np).to(torch.float64)
    
    # qe² = -0.5 Tr[A²]
    A_sq_trace = torch.einsum('nij,nji->n', A, A)
    qe2 = -0.5 * A_sq_trace
    
    # Q² = Q_ij Q_ij
    Q2 = torch.sum(Q * Q, dim=(-2, -1))
    
    qe2_np = qe2.cpu().numpy()
    Q2_np = Q2.cpu().numpy()
    
    # Standard deviations
    sigma_qe2 = np.std(qe2_np)
    sigma_Q2 = np.std(Q2_np)
    
    # Clip extremes for stable binning
    qe2_lo = np.percentile(qe2_np, clip_percentile)
    qe2_hi = np.percentile(qe2_np, 100 - clip_percentile)
    
    mask = (qe2_np >= qe2_lo) & (qe2_np <= qe2_hi)
    qe2_clipped = qe2_np[mask]
    Q2_clipped = Q2_np[mask]
    
    # Bin and compute conditional mean
    bin_edges = np.linspace(qe2_lo, qe2_hi, bins + 1)
    bin_idx = np.digitize(qe2_clipped, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    
    phi = np.zeros(bins)
    counts = np.zeros(bins, dtype=int)
    
    for i in range(bins):
        mask_bin = (bin_idx == i)
        if mask_bin.sum() > 0:
            phi[i] = Q2_clipped[mask_bin].mean()
            counts[i] = mask_bin.sum()
    
    # Normalize
    phi = phi / (sigma_qe2 * sigma_Q2 + EPS)
    
    # Plot only bins with sufficient samples
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = counts > 20
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers[valid], phi[valid], 'o-', lw=1.5, ms=4)
    ax.set_xlabel('$qe^2$')
    ax.set_ylabel('$\\phi(qe^2)$')
    ax.set_title('Fig. 12: $\\phi$ vs $qe^2$')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    fig.tight_layout()
    return fig

# ============ Main ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--velgrad', required=True, help='Path to velGrad.mat')
    parser.add_argument('--ph', required=True, help='Path to PH.mat (anisotropic Q)')
    parser.add_argument('--mapping', default='auto', choices=['auto', 'matlab_col', 'row_major'])
    parser.add_argument('--out', default='./figs_dns', help='Output directory')
    parser.add_argument('--bins', type=int, default=80, help='Histogram bins')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    print("Loading data...")
    V_A = load_mat(args.velgrad)
    V_Q = load_mat(args.ph)
    assert V_A.shape == V_Q.shape, f"Shape mismatch: A={V_A.shape}, Q={V_Q.shape}"
    
    # Detect mapping
    if args.mapping == 'auto':
        mapping = detect_mapping(V_A, V_Q)
    else:
        mapping = args.mapping
        print(f"Using mapping: {mapping}\n")
    
    # Reshape
    A_np = reshape_tensors(V_A, mapping)
    Q_np = reshape_tensors(V_Q, mapping)
    
    # Diagnostics
    tr_A = np.mean(np.abs(A_np[:, 0, 0] + A_np[:, 1, 1] + A_np[:, 2, 2]))
    tr_Q = np.mean(np.abs(Q_np[:, 0, 0] + Q_np[:, 1, 1] + Q_np[:, 2, 2]))
    print(f"Mean |Tr[A]|: {tr_A:.2e}  (expect ~0 for incompressible)")
    print(f"Mean |Tr[Q]|: {tr_Q:.2e}  (expect ~0 by construction)")
    print(f"Mean ||A||_F: {np.mean(np.linalg.norm(A_np.reshape(-1, 9), axis=1)):.3f}")
    print(f"Mean ||Q||_F: {np.mean(np.linalg.norm(Q_np.reshape(-1, 9), axis=1)):.3f}\n")
    
    # Generate figures
    print("Generating Fig. 9...")
    fig9 = fig9_alignment_s_Qhat0(A_np, Q_np, bins=args.bins)
    fig9.savefig(out_dir / 'fig9_dns.png', dpi=200)
    plt.close(fig9)
    
    print("Generating Fig. 10...")
    fig10 = fig10_alignment_vorticity_Qhat0(A_np, Q_np, bins=args.bins)
    fig10.savefig(out_dir / 'fig10_dns.png', dpi=200)
    plt.close(fig10)
    
    print("Generating Fig. 11...")
    fig11 = fig11_pdf_omega(A_np, Q_np, bins=args.bins)
    fig11.savefig(out_dir / 'fig11_dns.png', dpi=200)
    plt.close(fig11)
    
    print("Generating Fig. 12...")
    fig12 = fig12_phi_vs_qe2(A_np, Q_np, bins=60)
    fig12.savefig(out_dir / 'fig12_dns.png', dpi=200)
    plt.close(fig12)
    
    print(f"\n✓ Saved plots to {out_dir.resolve()}")

if __name__ == '__main__':
    main()