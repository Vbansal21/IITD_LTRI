#!/usr/bin/env python3
# new_figs.py
#
# Recreate Figs. 9–12 from:
#   "An augmented invariant-based model of the pressure Hessian tensor..."
#   (Phys. Fluids 35, 125124, 2023)
#
# FIXES for scaling/value issues:
#   • Headless-safe plotting (Agg backend) – no Qt/Wayland.
#   • Explicit handling of MATLAB vs row-major flattening for (9, N)/(N, 9) data.
#     -> Auto-detect mapping using incompressibility (trace≈0) + invariant bounds
#        and apply the same mapping to A (vel-grad) and PH (pressure Hessian).
#   • Exact normalizations from the paper:
#       e^2 = A_ij A_ij
#       a = A / e,  s = (a+a^T)/2,  w = (a-a^T)/2
#       Q0 = Q / e^2,  ω = ||Q0||_F,  Qhat0 = Q0 / ω
#       qe2 = -0.5 * Tr(A^2)         (2nd invariant of raw A)
#       φ(qe2) = E[ Q_ij Q_ij | qe2 ] / (σ_qe2 * σ_{Q_ij Q_ij})
#   • All square-roots use torch.sqrt with clamping (no pow(·, 0.5)).
#   • Chunked streaming for 1M+ samples; no full-matrix eigendecomp in RAM.
#
# Usage:
#   python new_figs.py --velgrad velGrad.mat --ph PH.mat --out ./figs \
#                      --bins 64 --chunk 131072 [--mapping auto|matlab_col|row_major]
#
# Outputs:
#   figs/fig9_align_s_vs_Qhat0.png
#   figs/fig10_align_vorticity_vs_Qhat0.png
#   figs/fig11_pdf_omega.png
#   figs/fig12_phi_vs_qe2.png
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe

from pathlib import Path
import argparse
import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt

# ---------- numerics ----------
_EPS = 1e-12

def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=_EPS))

def _sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))

def _antisym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M - M.transpose(-1, -2))

def _fro2(M: torch.Tensor) -> torch.Tensor:
    return torch.sum(M * M, dim=(-2, -1))

def normalized_tensors(A: torch.Tensor, Q: torch.Tensor):
    """
    e^2 = A_ij A_ij ; e = sqrt(e^2) ; a = A/e
    s = sym(a), w = antisym(a)
    Q0 = Q / e^2 ; ω = ||Q0||_F ; Qhat0 = Q0 / ω
    """
    A = A.to(torch.float64)
    Q = Q.to(torch.float64)

    Q = _sym(Q)  # enforce symmetry of anisotropic PH

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
    Symmetric 3x3 eigen-decomposition, descending eigenvalues.
    Returns (evals_sorted, evecs_sorted_as_columns)
    """
    evals, evecs = torch.linalg.eigh(M)                 # ascending
    idx = torch.argsort(evals, dim=-1, descending=True) # (...,3)
    evals_sorted = torch.gather(evals, -1, idx)
    gather_idx = idx.unsqueeze(-2).expand(-1, 3, -1)    # (...,3,3)
    evecs_sorted = torch.gather(evecs, -1, gather_idx)
    return evals_sorted, evecs_sorted

def vorticity_from_w(w: torch.Tensor):
    """
    x_i = (ε_ijk w_jk)/2  (unit vector)
    """
    x = torch.stack([
        (w[:, 2, 1] - w[:, 1, 2]) * 0.5,
        (w[:, 0, 2] - w[:, 2, 0]) * 0.5,
        (w[:, 1, 0] - w[:, 0, 1]) * 0.5
    ], dim=1)
    return torch.nn.functional.normalize(x, dim=-1, eps=1e-15)

# ---------- histogram & stats (streaming) ----------
class OnlineHist:
    def __init__(self, edges: np.ndarray):
        self.edges = edges
        self.counts = np.zeros(len(edges) - 1, dtype=np.float64)
        self.total = 0

    def update(self, values: np.ndarray):
        if values.size == 0:
            return
        c, _ = np.histogram(values, bins=self.edges)
        self.counts += c
        self.total += values.size

    def pdf(self):
        binw = np.diff(self.edges)
        denom = self.total * binw
        denom[denom == 0] = np.inf
        return (self.edges[:-1] + self.edges[1:]) * 0.5, self.counts / denom

def welford_update(n, mean, M2, x: np.ndarray):
    for v in x:
        n1 = n + 1
        delta = v - mean
        mean += delta / n1
        M2 += delta * (v - mean)
        n = n1
    return n, mean, M2

# ---------- loader with mapping autodetect ----------
def _load_mat_anykey(path: str) -> np.ndarray:
    mat = scipy.io.loadmat(path)
    key = next(k for k in mat if k not in ('__header__', '__version__', '__globals__'))
    return np.asarray(mat[key])

def _to_N9(arr: np.ndarray) -> np.ndarray:
    # Accept (9,N) or (N,9) → return (N,9)
    if arr.ndim != 2 or 9 not in arr.shape:
        raise ValueError(f"Expected a 2D array with one dimension==9, got {arr.shape}")
    return arr.T if arr.shape[0] == 9 else arr

def _reconstruct_tensor_N33(V: np.ndarray, mapping: str) -> np.ndarray:
    """
    V: (N,9)
    mapping:
      - 'matlab_col': v = [A11, A21, A31, A12, A22, A32, A13, A23, A33] (MATLAB M(:))
      - 'row_major' : v = [A11, A12, A13, A21, A22, A23, A31, A32, A33]
    """
    N = V.shape[0]
    A = np.empty((N, 3, 3), dtype=np.float64)
    if mapping == 'matlab_col':
        A[:, 0, 0] = V[:, 0]; A[:, 1, 0] = V[:, 1]; A[:, 2, 0] = V[:, 2]
        A[:, 0, 1] = V[:, 3]; A[:, 1, 1] = V[:, 4]; A[:, 2, 1] = V[:, 5]
        A[:, 0, 2] = V[:, 6]; A[:, 1, 2] = V[:, 7]; A[:, 2, 2] = V[:, 8]
    elif mapping == 'row_major':
        A[:, 0, 0] = V[:, 0]; A[:, 0, 1] = V[:, 1]; A[:, 0, 2] = V[:, 2]
        A[:, 1, 0] = V[:, 3]; A[:, 1, 1] = V[:, 4]; A[:, 1, 2] = V[:, 5]
        A[:, 2, 0] = V[:, 6]; A[:, 2, 1] = V[:, 7]; A[:, 2, 2] = V[:, 8]
    else:
        raise ValueError("mapping must be 'matlab_col' or 'row_major'")
    return A

def _score_mapping(A33: np.ndarray) -> float:
    """
    Lower score => mapping more plausible.
    Combines: incompressibility (trace≈0) + invariant bounds for a.
    """
    # incompressibility proxy
    tr = A33[:, 0, 0] + A33[:, 1, 1] + A33[:, 2, 2]
    tr_rmse = np.sqrt(np.mean(tr * tr))

    # invariants for a = A / ||A||
    A = torch.from_numpy(A33)
    e2 = _fro2(A).numpy()
    e  = np.sqrt(np.clip(e2, _EPS, None))
    a  = (A.numpy() / e[:, None, None]).astype(np.float64)
    a = torch.from_numpy(a)
    a2 = torch.einsum('bij,bjk->bik', a, a).numpy()
    tr_a2 = a2[:, 0, 0] + a2[:, 1, 1] + a2[:, 2, 2]
    q = -0.5 * tr_a2  # for incompressible p=0
    # count how many far outside theoretical bounds [-0.5, 1]
    outside = np.mean((q < -0.6) | (q > 1.1))
    # combined score: prioritize incompressibility, penalize bound violations
    return tr_rmse + 10.0 * outside

def autodetect_mapping(VA: np.ndarray, VP: np.ndarray, sample: int = 200_000) -> str:
    """
    Try both mappings on a subsample; pick the one with lower score.
    """
    N = VA.shape[0]
    idx = np.arange(N) if N <= sample else np.random.default_rng(123).choice(N, size=sample, replace=False)

    def score(name):
        A33 = _reconstruct_tensor_N33(VA[idx], name)
        return _score_mapping(A33)

    s_col = score('matlab_col')
    s_row = score('row_major')
    mapping = 'matlab_col' if s_col <= s_row else 'row_major'
    print(f"[auto] mapping scores → matlab_col: {s_col:.3e}, row_major: {s_row:.3e}  => using {mapping}")
    # Apply same mapping to PH as well
    return mapping

# ---------- figure 9 ----------
def compute_fig9_chunked(A: torch.Tensor, Q: torch.Tensor, bins: int, chunk: int):
    edges = np.linspace(0.0, 1.0, bins + 1)
    H = {(i, j): OnlineHist(edges.copy()) for i in range(3) for j in range(3)}

    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        _, s, _, _, Qhat0, _ = normalized_tensors(A[start:end], Q[start:end])
        _, Es = eigh_sorted_desc(s)
        _, Ep = eigh_sorted_desc(Qhat0)

        Es = torch.nn.functional.normalize(Es, dim=-2, eps=1e-15)
        Ep = torch.nn.functional.normalize(Ep, dim=-2, eps=1e-15)
        COS = torch.matmul(Es.transpose(-1, -2), Ep).abs().cpu().numpy()  # (b,3,3)

        for i in range(3):
            for j in range(3):
                H[(i, j)].update(COS[:, i, j])

    return {k: H[k].pdf() for k in H}

def plot_fig9(results, save_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.2), constrained_layout=True)
    row_names = ['as', 'bs', 'cs']
    col_names = ['ap', 'bp', 'cp']
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

# ---------- figure 10 ----------
def compute_fig10_chunked(A: torch.Tensor, Q: torch.Tensor, bins: int, chunk: int):
    edges = np.linspace(0.0, 1.0, bins + 1)
    H = {j: OnlineHist(edges.copy()) for j in range(3)}  # ap,bp,cp

    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        _, _, w, _, Qhat0, _ = normalized_tensors(A[start:end], Q[start:end])
        vhat = vorticity_from_w(w)  # (b,3)
        _, Ep = eigh_sorted_desc(Qhat0)
        Ep = torch.nn.functional.normalize(Ep, dim=-2, eps=1e-15)
        vhat = torch.nn.functional.normalize(vhat, dim=-1, eps=1e-15)

        COS = torch.abs(torch.einsum('bi,bij->bj', vhat, Ep)).cpu().numpy()
        for j in range(3):
            H[j].update(COS[:, j])

    return {j: H[j].pdf() for j in range(3)}

def plot_fig10(results, save_path: Path):
    names = ['ap', 'bp', 'cp']
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

# ---------- figure 11 ----------
def compute_fig11_chunked(A: torch.Tensor, Q: torch.Tensor, bins: int, chunk: int, sample_cap: int = 200_000, clip_hi=99.5):
    rng = np.random.default_rng(123)
    sample = np.array([], dtype=np.float64)

    # pass 1: sampling for percentile range
    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        _, _, _, _, _, omega = normalized_tensors(A[start:end], Q[start:end])
        om = omega.cpu().numpy()
        if sample.size < sample_cap:
            take = min(sample_cap - sample.size, om.size)
            sample = np.concatenate([sample, om[:take]])
            if take < om.size:
                idx = rng.integers(0, sample_cap, size=om.size - take)
                sample[idx] = om[take:]
        else:
            idx = rng.integers(0, sample_cap, size=om.size)
            sample[idx] = om

    lo = float(np.min(sample)) if sample.size else 0.0
    hi = float(np.percentile(sample, clip_hi)) if sample.size else 1.0
    edges = np.linspace(lo, hi, bins + 1)
    H = OnlineHist(edges.copy())

    # pass 2: histogram
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        _, _, _, _, _, omega = normalized_tensors(A[start:end], Q[start:end])
        om = omega.cpu().numpy()
        om = om[(om >= lo) & (om <= hi)]
        H.update(om)

    return H.pdf()

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

# ---------- figure 12 ----------
def compute_qe2_Q2(A: torch.Tensor, Q: torch.Tensor):
    # qe2 = -0.5 * Tr(A^2); Q2 = sum(Q^2) with Q being anisotropic PH (raw, not normalized)
    AtA = torch.einsum('bij,bji->b', A, A)
    qe2 = -0.5 * AtA
    Q2  = _fro2(Q.to(torch.float64))
    return qe2.to(torch.float64), Q2

def compute_fig12_chunked(A: torch.Tensor, Q: torch.Tensor, nbins: int, chunk: int, clip_percent: float = 0.5):
    rng = np.random.default_rng(321)
    sample_cap = 200_000
    sample_q = np.array([], dtype=np.float64)

    # pass 1: get σ(qe2), σ(Q2) + percentile bounds for qe2
    n_q, mean_q, M2_q = 0, 0.0, 0.0
    n_Q2, mean_Q2, M2_Q2 = 0, 0.0, 0.0

    N = A.shape[0]
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        qe2, Q2 = compute_qe2_Q2(A[start:end].to(torch.float64), Q[start:end].to(torch.float64))
        q_np  = qe2.cpu().numpy()
        Q2_np = Q2.cpu().numpy()

        n_q,  mean_q,  M2_q  = welford_update(n_q,  mean_q,  M2_q,  q_np)
        n_Q2, mean_Q2, M2_Q2 = welford_update(n_Q2, mean_Q2, M2_Q2, Q2_np)

        # reservoir sample for percentiles
        if sample_q.size < sample_cap:
            take = min(sample_cap - sample_q.size, q_np.size)
            sample_q = np.concatenate([sample_q, q_np[:take]])
            if take < q_np.size:
                idx = rng.integers(0, sample_cap, size=q_np.size - take)
                sample_q[idx] = q_np[take:]
        else:
            idx = rng.integers(0, sample_cap, size=q_np.size)
            sample_q[idx] = q_np

    std_q  = np.sqrt(M2_q / max(n_q - 1, 1))
    std_Q2 = np.sqrt(M2_Q2 / max(n_Q2 - 1, 1))
    denom = max(std_q * std_Q2, 1e-15)

    lo = float(np.percentile(sample_q, clip_percent)) if sample_q.size else 0.0
    hi = float(np.percentile(sample_q, 100.0 - clip_percent)) if sample_q.size else 1.0
    edges = np.linspace(lo, hi, nbins + 1)

    sums_Q2 = np.zeros(nbins, dtype=np.float64)
    counts  = np.zeros(nbins, dtype=np.int64)

    # pass 2: conditional means
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        qe2, Q2 = compute_qe2_Q2(A[start:end].to(torch.float64), Q[start:end].to(torch.float64))
        q_np  = qe2.cpu().numpy()
        Q2_np = Q2.cpu().numpy()

        mask = (q_np >= lo) & (q_np <= hi)
        if not np.any(mask):
            continue
        q_np  = q_np[mask]
        Q2_np = Q2_np[mask]

        idx = np.digitize(q_np, edges) - 1
        valid = (idx >= 0) & (idx < nbins)
        if not np.any(valid):
            continue
        idx = idx[valid]
        Q2_np = Q2_np[valid]
        np.add.at(sums_Q2, idx, Q2_np)
        np.add.at(counts,  idx, 1)

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

# ---------- main ----------
def main(velgrad_mat: str, ph_mat: str, out_dir: str, bins: int, chunk: int, mapping: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {velgrad_mat} and {ph_mat} ...")
    V_A = _to_N9(_load_mat_anykey(velgrad_mat))
    V_P = _to_N9(_load_mat_anykey(ph_mat))

    if mapping == 'auto':
        mapping = autodetect_mapping(V_A, V_P, sample=200_000)
    else:
        assert mapping in ('matlab_col', 'row_major'), "mapping must be auto|matlab_col|row_major"
        print(f"[cfg] using mapping={mapping}")

    # reconstruct tensors
    A_np = _reconstruct_tensor_N33(V_A, mapping)
    P_np = _reconstruct_tensor_N33(V_P, mapping)

    # anisotropic Q (trace-free)
    trP = P_np[:, 0, 0] + P_np[:, 1, 1] + P_np[:, 2, 2]
    I3  = np.eye(3, dtype=np.float64)[None, :, :]
    Q_np = P_np - (trP[:, None, None] / 3.0) * I3

    # quick diagnostics (helpful for scale sanity)
    trA_rmse = float(np.sqrt(np.mean((A_np[:, 0, 0] + A_np[:, 1, 1] + A_np[:, 2, 2]) ** 2)))
    e2_mean  = float(np.mean(np.sum(A_np * A_np, axis=(1, 2))))
    Q2_mean  = float(np.mean(np.sum(Q_np * Q_np, axis=(1, 2))))
    print(f"[diag] trace(A) RMSE ≈ {trA_rmse:.3e} (should be small for incompressible)")
    print(f"[diag] mean(e^2) ≈ {e2_mean:.3e}, mean(||Q||_F^2) ≈ {Q2_mean:.3e}")

    # torch tensors
    A = torch.from_numpy(A_np)
    Q = torch.from_numpy(Q_np)

    # Fig. 9
    res9 = compute_fig9_chunked(A, Q, bins=bins, chunk=chunk)
    plot_fig9(res9, out / "fig9_align_s_vs_Qhat0.png")

    # Fig. 10
    res10 = compute_fig10_chunked(A, Q, bins=bins, chunk=chunk)
    plot_fig10(res10, out / "fig10_align_vorticity_vs_Qhat0.png")

    # Fig. 11
    c11, h11 = compute_fig11_chunked(A, Q, bins=bins, chunk=chunk, sample_cap=200_000, clip_hi=99.5)
    plot_fig11(c11, h11, out / "fig11_pdf_omega.png")

    # Fig. 12
    c12, phi12, cnt12 = compute_fig12_chunked(A, Q, nbins=max(40, bins), chunk=chunk, clip_percent=0.5)
    plot_fig12(c12, phi12, cnt12, out / "fig12_phi_vs_qe2.png", min_count=50)

    print(f"[OK] Saved: {out.resolve()}/fig9_align_s_vs_Qhat0.png")
    print(f"[OK] Saved: {out.resolve()}/fig10_align_vorticity_vs_Qhat0.png")
    print(f"[OK] Saved: {out.resolve()}/fig11_pdf_omega.png")
    print(f"[OK] Saved: {out.resolve()}/fig12_phi_vs_qe2.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Recreate Figs. 9–12 from velGrad (A) and PH (pressure Hessian).")
    ap.add_argument("--velgrad", required=True, help="Path to velGrad.mat (shape 9xN or Nx9)")
    ap.add_argument("--ph",       required=True, help="Path to PH.mat (shape 9xN or Nx9)")
    ap.add_argument("--out",      default="figs", help="Output directory")
    ap.add_argument("--bins",     type=int, default=64, help="Histogram bins")
    ap.add_argument("--chunk",    type=int, default=131072, help="Chunk size for streaming stats")
    ap.add_argument("--mapping",  default="auto", choices=["auto", "matlab_col", "row_major"],
                    help="Flattening convention of 9-long vectors.")
    args = ap.parse_args()
    main(args.velgrad, args.ph, out_dir=args.out, bins=args.bins, chunk=args.chunk, mapping=args.mapping)
