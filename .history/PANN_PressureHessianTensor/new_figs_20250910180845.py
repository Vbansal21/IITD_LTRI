#!/usr/bin/env python3
# new_figs.py
#
# Recreate Figs. 9–12 robustly from velGrad.mat (A) and PH.mat (pressure Hessian, trace removed here).
# - Headless-safe (uses Agg backend; no Qt/Wayland needed)
# - No fragile LaTeX in titles/labels
# - Numerically stable (sqrt, clamping, explicit symmetrization)
# - Chunked processing for large N (default: 1,048,576)
#
# Usage:
#   python new_figs.py --velgrad velGrad.mat --ph PH.mat --out ./figs --bins 64 --chunk 131072
#
# Outputs in --out:
#   fig9_align_s_vs_Qhat0.png
#   fig10_align_vorticity_vs_Qhat0.png
#   fig11_pdf_omega.png
#   fig12_phi_vs_qe2.png
#
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe

from pathlib import Path
import argparse
import numpy as np
import torch
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

# ------------------------------
# Data loader (as provided, with a tiny dtype tweak for stability)
# ------------------------------
class MatlabDataset(torch.utils.data.Dataset):
    def __init__(self, vel_grad_path, pressure_hessian_path):
        print(f"Loading data from {vel_grad_path} and {pressure_hessian_path}...")

        def load_mat_data(path):
            mat = scipy.io.loadmat(path)
            key = next(k for k in mat if k not in ('__header__', '__version__', '__globals__'))
            return mat[key].astype(np.float32)

        vel_grad_data = load_mat_data(vel_grad_path)
        if vel_grad_data.shape[0] == 9:
            vel_grad_data = vel_grad_data.T
        self.A = torch.from_numpy(vel_grad_data).view(-1, 3, 3)

        ph_data = load_mat_data(pressure_hessian_path)
        if ph_data.shape[0] == 9:
            ph_data = ph_data.T
        raw_P = torch.from_numpy(ph_data).view(-1, 3, 3)

        # anisotropic, trace-free Q
        trace_P = torch.einsum('bii->b', raw_P).unsqueeze(-1).unsqueeze(-1)
        self.Q = raw_P - (trace_P / 3.0) * torch.eye(3, dtype=raw_P.dtype).unsqueeze(0)

        assert self.A.shape[0] == self.Q.shape[0], "Data sample counts do not match."
        self.num_samples = self.A.shape[0]
        print(f"Data loaded successfully. Found {self.num_samples} samples.")

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.A[idx], self.Q[idx]
    def get_full_dataset(self): return self.A, self.Q


# ------------------------------
# Numerics
# ------------------------------
_EPS = 1e-12

def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    # Use sqrt; avoid pow(·, 0.5) due to precision oddities
    return torch.sqrt(torch.clamp(x, min=_EPS))

def _sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))

def _antisym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M - M.transpose(-1, -2))

def _fro2(M: torch.Tensor) -> torch.Tensor:
    return torch.sum(M * M, dim=(-2, -1))

def normalized_tensors(A: torch.Tensor, Q: torch.Tensor):
    """
    Compute normalized objects used by the figures:
      e^2 = A_ij A_ij ; e = sqrt(e^2)
      a = A / e
      s = sym(a), w = antisym(a)
      Q0 = Q / e^2
      omega = ||Q0||_F
      Qhat0 = Q0 / omega
    All operations are float64; sqrt is clamped.
    """
    A = A.to(torch.float64)
    Q = Q.to(torch.float64)

    Q = _sym(Q)  # enforce symmetry

    e2 = _fro2(A)                        # (B,)
    e  = _safe_sqrt(e2)                  # (B,)
    a  = A / e.view(-1, 1, 1)            # (B,3,3)
    s  = _sym(a)
    w  = _antisym(a)

    Q0    = Q / e2.view(-1, 1, 1)        # (B,3,3)
    omega = _safe_sqrt(_fro2(Q0))        # (B,)
    Qhat0 = Q0 / omega.view(-1, 1, 1)    # (B,3,3)

    return a, s, w, Q0, Qhat0, omega

def eigh_sorted_desc(M: torch.Tensor):
    """
    Eigen-decomposition for symmetric 3x3 matrices.
    Returns eigenvalues (desc) and eigenvectors (columns) sorted by descending eigenvalue.
    """
    evals, evecs = torch.linalg.eigh(M)                 # ascending
    idx = torch.argsort(evals, dim=-1, descending=True) # (...,3)
    evals_sorted = torch.gather(evals, -1, idx)

    gather_idx = idx.unsqueeze(-2).expand(-1, 3, -1)    # (...,3,3)
    evecs_sorted = torch.gather(evecs, -1, gather_idx)  # (...,3,3) columns aligned with evals_sorted
    return evals_sorted, evecs_sorted

def vorticity_from_w(w: torch.Tensor):
    """
    Vorticity direction from antisymmetric part w of a:
      x_i = (ε_ijk w_jk)/2
    Returns a unit vector (B,3).
    """
    x = torch.stack([
        (w[:, 2, 1] - w[:, 1, 2]) * 0.5,
        (w[:, 0, 2] - w[:, 2, 0]) * 0.5,
        (w[:, 1, 0] - w[:, 0, 1]) * 0.5
    ], dim=1)
    xn = torch.nn.functional.normalize(x, dim=-1, eps=1e-15)
    return xn

# ------------------------------
# Incremental histogram helpers
# ------------------------------
class OnlineHist:
    def __init__(self, edges: np.ndarray):
        self.edges = edges
        self.counts = np.zeros(len(edges) - 1, dtype=np.float64)
        self.total = 0

    def update(self, values: np.ndarray):
        # expects 1D array
        c, _ = np.histogram(values, bins=self.edges)
        self.counts += c
        self.total += values.size

    def pdf(self):
        # density=True equivalent
        binw = np.diff(self.edges)
        denom = self.total * binw
        denom[denom == 0] = np.inf
        return (self.edges[:-1] + self.edges[1:]) * 0.5, self.counts / denom

def online_mean_std(prev_n, prev_mean, prev_M2, new_values):
    """
    Welford update for streaming mean/std.
    Returns updated (n, mean, M2). std = sqrt(M2/(n-1)).
    """
    n = prev_n
    mean = prev_mean
    M2 = prev_M2
    for x in new_values:
        n1 = n + 1
        delta = x - mean
        mean += delta / n1 if n1 > 0 else 0.0
        delta2 = x - mean
        M2 += delta * delta2
        n = n1
    return n, mean, M2

# ------------------------------
# Figure 9 (alignment PDFs between eigenvectors of s and Qhat0)
# ------------------------------
def compute_fig9_chunked(A: torch.Tensor, Q: torch.Tensor, bins: int, chunk: int):
    edges = np.linspace(0.0, 1.0, bins + 1)
    # 9 histograms for (i,j) in (0..2, 0..2)
    H = {(i, j): OnlineHist(edges.copy()) for i in range(3) for j in range(3)}

    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        A_c = A[start:end]
        Q_c = Q[start:end]

        _, s, _, _, Qhat0, _ = normalized_tensors(A_c, Q_c)
        _, Es = eigh_sorted_desc(s)
        _, Ep = eigh_sorted_desc(Qhat0)

        # Normalize columns and compute absolute direction cosines
        Es = torch.nn.functional.normalize(Es, dim=-2, eps=1e-15)
        Ep = torch.nn.functional.normalize(Ep, dim=-2, eps=1e-15)
        COS = torch.matmul(Es.transpose(-1, -2), Ep).abs().cpu().numpy()  # (b,3,3)

        for i in range(3):
            for j in range(3):
                H[(i, j)].update(COS[:, i, j])

    # Convert to PDFs
    results = {}
    for i in range(3):
        for j in range(3):
            centers, pdf_vals = H[(i, j)].pdf()
            results[(i, j)] = (centers, pdf_vals)
    return results

def plot_fig9(results, save_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.2), constrained_layout=True)
    row_names = ['as','bs','cs']  # eigenvectors of s, descending order
    col_names = ['ap','bp','cp']  # eigenvectors of Qhat0, descending order
    for i in range(3):
        for j in range(3):
            centers, hist = results[(i, j)]
            ax = axes[i, j]
            ax.plot(centers, hist, lw=1.6)
            ax.set_xlim(0, 1)
            ax.set_xlabel('|cos(theta)|', fontsize=10)
            ax.set_ylabel('PDF', fontsize=10)
            ax.set_title(f'{row_names[i]}(s) vs {col_names[j]}(Qhat0)', fontsize=10)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Fig. 9: Alignment PDFs — s eigenvectors vs Qhat0 eigenvectors', fontsize=12)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ------------------------------
# Figure 10 (vorticity alignment vs eigenvectors of Qhat0)
# ------------------------------
def compute_fig10_chunked(A: torch.Tensor, Q: torch.Tensor, bins: int, chunk: int):
    edges = np.linspace(0.0, 1.0, bins + 1)
    H = {j: OnlineHist(edges.copy()) for j in range(3)}  # ap,bp,cp

    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        A_c = A[start:end]
        Q_c = Q[start:end]

        a, _, w, _, Qhat0, _ = normalized_tensors(A_c, Q_c)
        vhat = vorticity_from_w(w)  # (b,3)
        _, Ep = eigh_sorted_desc(Qhat0)
        Ep = torch.nn.functional.normalize(Ep, dim=-2, eps=1e-15)
        vhat = torch.nn.functional.normalize(vhat, dim=-1, eps=1e-15)

        # (b,3): absolute projection onto eigenvectors ap,bp,cp
        COS = torch.abs(torch.einsum('bi,bij->bj', vhat, Ep)).cpu().numpy()
        for j in range(3):
            H[j].update(COS[:, j])

    results = {}
    for j in range(3):
        centers, pdf_vals = H[j].pdf()
        results[j] = (centers, pdf_vals)
    return results

def plot_fig10(results, save_path: Path):
    names = ['ap','bp','cp']
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for j, ax in enumerate(axes):
        centers, hist = results[j]
        ax.plot(centers, hist, lw=1.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel('|cos(theta)|', fontsize=10)
        ax.set_ylabel('PDF', fontsize=10)
        ax.set_title(f'vorticity vs {names[j]}(Qhat0)', fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Fig. 10: Alignment PDFs — vorticity vs Qhat0 eigenvectors', fontsize=12)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ------------------------------
# Figure 11 (PDF of omega = ||Q0||_F)
#   Two-pass: percentile range via reservoir sample, then histogram
# ------------------------------
def reservoir_sample_append(reservoir: np.ndarray, cap: int, new_vals: np.ndarray, rng: np.random.Generator):
    if reservoir.size < cap:
        space = cap - reservoir.size
        take = min(space, new_vals.size)
        if take > 0:
            reservoir = np.concatenate([reservoir, new_vals[:take]])
        if take < new_vals.size:
            # replace uniformly at random among cap slots
            remain = new_vals[take:]
            idx = rng.integers(0, cap, size=remain.size)
            reservoir[idx] = remain
    else:
        idx = rng.integers(0, cap, size=new_vals.size)
        reservoir[idx] = new_vals
    return reservoir

def compute_fig11_chunked(A: torch.Tensor, Q: torch.Tensor, bins: int, chunk: int, sample_cap: int = 200_000, clip_hi=99.5):
    rng = np.random.default_rng(123)
    sample = np.array([], dtype=np.float64)

    # pass 1: build reservoir sample for percentiles
    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        _, _, _, _, _, omega = normalized_tensors(A[start:end], Q[start:end])
        omega_np = omega.cpu().numpy()
        sample = reservoir_sample_append(sample, sample_cap, omega_np, rng)

    lo = sample.min() if sample.size else 0.0
    hi = np.percentile(sample, clip_hi) if sample.size else 1.0
    edges = np.linspace(lo, hi, bins + 1)
    H = OnlineHist(edges.copy())

    # pass 2: histogram
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        _, _, _, _, _, omega = normalized_tensors(A[start:end], Q[start:end])
        omega_np = omega.cpu().numpy()
        omega_np = omega_np[(omega_np >= lo) & (omega_np <= hi)]
        if omega_np.size:
            H.update(omega_np)

    centers, pdf_vals = H.pdf()
    return centers, pdf_vals

def plot_fig11(centers, hist, save_path: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.plot(centers, hist, lw=1.8)
    ax.set_xlabel('omega = ||Q0||_F', fontsize=10)
    ax.set_ylabel('PDF', fontsize=10)
    ax.set_title('Fig. 11: PDF of omega', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ------------------------------
# Figure 12 (phi(q e^2) vs q e^2)
#   phi(qe2) = E[Q_{mn}Q_{mn} | qe2] / (std(qe2) * std(Q_{mn}Q_{mn}))
#   Two-pass: percentiles for qe2-range + stds (Welford), then conditional means
# ------------------------------
def compute_qe2_Q2(A: torch.Tensor, Q: torch.Tensor):
    # qe2 = -0.5 * Tr(A^2); Q2 = sum(Q^2)
    AtA = torch.einsum('bij,bji->b', A, A)
    qe2 = -0.5 * AtA
    Q2 = _fro2(Q.to(torch.float64))
    return qe2.to(torch.float64), Q2

def compute_fig12_chunked(A: torch.Tensor, Q: torch.Tensor, nbins: int, chunk: int, clip_percent: float = 0.5):
    rng = np.random.default_rng(321)
    sample_cap = 200_000
    sample_q = np.array([], dtype=np.float64)

    # pass 1: percentiles for qe2 range; also stds via Welford
    n_q, mean_q, M2_q = 0, 0.0, 0.0
    n_Q2, mean_Q2, M2_Q2 = 0, 0.0, 0.0

    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        qe2, Q2 = compute_qe2_Q2(A[start:end].to(torch.float64), Q[start:end].to(torch.float64))
        q_np = qe2.cpu().numpy()
        Q2_np = Q2.cpu().numpy()

        # Welford updates
        n_q, mean_q, M2_q = online_mean_std(n_q, mean_q, M2_q, q_np)
        n_Q2, mean_Q2, M2_Q2 = online_mean_std(n_Q2, mean_Q2, M2_Q2, Q2_np)

        sample_q = reservoir_sample_append(sample_q, sample_cap, q_np, rng)

    std_q = np.sqrt(M2_q / max(n_q - 1, 1))
    std_Q2 = np.sqrt(M2_Q2 / max(n_Q2 - 1, 1))
    denom = max(std_q * std_Q2, 1e-15)

    if sample_q.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = np.percentile(sample_q, clip_percent)
        hi = np.percentile(sample_q, 100.0 - clip_percent)

    edges = np.linspace(lo, hi, nbins + 1)
    sums_Q2 = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    # pass 2: conditional mean of Q2 per qe2 bin
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        qe2, Q2 = compute_qe2_Q2(A[start:end].to(torch.float64), Q[start:end].to(torch.float64))
        q_np = qe2.cpu().numpy()
        Q2_np = Q2.cpu().numpy()

        mask = (q_np >= lo) & (q_np <= hi)
        if not np.any(mask):
            continue
        q_np = q_np[mask]
        Q2_np = Q2_np[mask]

        idx = np.digitize(q_np, edges) - 1
        valid = (idx >= 0) & (idx < nbins)
        if not np.any(valid):
            continue
        idx = idx[valid]
        Q2_np = Q2_np[valid]
        # accumulate sums and counts
        np.add.at(sums_Q2, idx, Q2_np)
        np.add.at(counts, idx, 1)

    phi = np.full(nbins, np.nan, dtype=np.float64)
    ok = counts > 0
    phi[ok] = (sums_Q2[ok] / counts[ok]) / denom
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, phi, counts

def plot_fig12(centers, phi, counts, save_path: Path, min_count=50):
    mask = (counts >= min_count) & np.isfinite(phi)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(centers[mask], phi[mask], lw=1.8)
    ax.set_xlabel('q * e^2', fontsize=10)
    ax.set_ylabel('phi(q * e^2)', fontsize=10)
    ax.set_title('Fig. 12: phi(q * e^2) vs q * e^2', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ------------------------------
# Entrypoint
# ------------------------------
def main(velgrad_mat: str, ph_mat: str, out_dir: str, bins: int, chunk: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = MatlabDataset(velgrad_mat, ph_mat)
    A, Q = ds.get_full_dataset()

    # Figure 9
    res9 = compute_fig9_chunked(A, Q, bins=bins, chunk=chunk)
    plot_fig9(res9, out / "fig9_align_s_vs_Qhat0.png")

    # Figure 10
    res10 = compute_fig10_chunked(A, Q, bins=bins, chunk=chunk)
    plot_fig10(res10, out / "fig10_align_vorticity_vs_Qhat0.png")

    # Figure 11
    c11, h11 = compute_fig11_chunked(A, Q, bins=bins, chunk=chunk, sample_cap=200_000, clip_hi=99.5)
    plot_fig11(c11, h11, out / "fig11_pdf_omega.png")

    # Figure 12
    c12, phi12, cnt12 = compute_fig12_chunked(A, Q, nbins=max(40, bins), chunk=chunk, clip_percent=0.5)
    plot_fig12(c12, phi12, cnt12, out / "fig12_phi_vs_qe2.png", min_count=50)

    print(f"[OK] Saved: {out.resolve()}/fig9_align_s_vs_Qhat0.png")
    print(f"[OK] Saved: {out.resolve()}/fig10_align_vorticity_vs_Qhat0.png")
    print(f"[OK] Saved: {out.resolve()}/fig11_pdf_omega.png")
    print(f"[OK] Saved: {out.resolve()}/fig12_phi_vs_qe2.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recreate Figs. 9–12 from A (vel-grad) and Q (anisotropic PH).")
    parser.add_argument("--velgrad", required=True, help="Path to velGrad.mat (9xN or Nx9)")
    parser.add_argument("--ph", required=True, help="Path to PH.mat (9xN or Nx9)")
    parser.add_argument("--out", default="figs", help="Output directory for figures")
    parser.add_argument("--bins", type=int, default=64, help="Histogram bins")
    parser.add_argument("--chunk", type=int, default=131072, help="Chunk size for streaming computations")
    args = parser.parse_args()
    main(args.velgrad, args.ph, out_dir=args.out, bins=args.bins, chunk=args.chunk)
