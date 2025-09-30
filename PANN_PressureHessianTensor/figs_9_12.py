"""
DNS-only Figures 9–12 (from scratch, self-contained).

Reads velocity gradient (velGrad.mat) and pressure Hessian (PH.mat),
reconstructs 3×3 tensors from 9-component vectors with robust mapping
selection, and generates:
  - Fig 9: Alignment PDFs of S vs Q̂′ eigenvectors (3×3 grid)
  - Fig 10: Alignment PDFs of ω vs Q̂′ eigenvectors (1×3)
  - Fig 11: PDF of ψ = ||Q′||_F
  - Fig 12: φ(x) vs x with x = (q·ε²)/σ(q·ε²), φ(x) = ⟨QmnQmn | x-bin⟩·σ(x)/σ(QmnQmn)

Core definitions (paper-consistent):
  - ε = ||A||_F = sqrt(A:A)
  - a = A / ε
  - s = 0.5(a + aᵀ), w = 0.5(a − aᵀ)
  - q = −½ tr(a²), r = −⅓ tr(a³)
  - Q (anisotropic) = PH − tr(PH)/3 · I
  - Q′ = Q / ε², ψ = ||Q′||_F, Q̂′ = Q′/ψ

Notes:
  - Uses sqrt everywhere (no pow) to avoid precision traps.
  - Chooses 9→3×3 mapping via physics: A ~ incompressible; P ~ symmetric.
  - No dependency on other project files; safe to run standalone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# .MAT loading and 9→3×3 reconstruction
# -----------------------------------------------------------------------------

def load_mat_array(path: str, prefer: Tuple[str, ...]) -> np.ndarray:
    mat = scipy.io.loadmat(path)
    # Prefer explicit var names if present
    for name in prefer:
        if name in mat and mat[name].size > 0:
            return np.asarray(mat[name])
    # Fallback: first non-meta array
    for k, v in mat.items():
        if not k.startswith('__') and np.asarray(v).size > 0:
            return np.asarray(v)
    raise ValueError(f"No usable array found in {path}")


def _apply_map_2d(X: np.ndarray, map_name: str) -> np.ndarray:
    """Map a 2D (N,9) or (9,N) array to (N,3,3) using a specific layout.

    row  : [[v0 v1 v2],[v3 v4 v5],[v6 v7 v8]] (C-order)
    col  : [[v0 v3 v6],[v1 v4 v7],[v2 v5 v8]] (Fortran A(:))
    rowT : row then transpose
    colT : col then transpose
    """
    assert map_name in ('row', 'col', 'rowT', 'colT')
    Z = np.asarray(X)
    if Z.ndim != 2:
        raise ValueError(f"Expected 2D array with 9 components, got {Z.shape}")
    if Z.shape[0] == 9 and Z.shape[1] != 9:
        Z = Z.T
    if Z.shape[1] != 9:
        raise ValueError(f"Second dim must be 9, got {Z.shape}")
    if map_name == 'row':
        return Z.reshape(-1, 3, 3, order='C')
    if map_name == 'col':
        return Z.reshape(-1, 3, 3, order='F')
    if map_name == 'rowT':
        M = Z.reshape(-1, 3, 3, order='C')
        return np.swapaxes(M, -1, -2)
    if map_name == 'colT':
        M = Z.reshape(-1, 3, 3, order='F')
        return np.swapaxes(M, -1, -2)
    raise AssertionError('unreachable')


def reconstruct_3x3(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Return a dict of candidate (N,3,3) tensors for X of shapes (N,9), (9,N), (N,3,3), (3,3,N)."""
    X = np.asarray(X)
    out: Dict[str, np.ndarray] = {}
    if X.ndim == 3 and X.shape[-2:] == (3, 3):
        out['as_is'] = X.reshape(-1, 3, 3)
        return out
    if X.ndim == 3 and X.shape[:2] == (3, 3):
        out['swap012'] = np.swapaxes(X, 0, 2).swapaxes(0, 1).reshape(-1, 3, 3)
        return out
    if X.ndim != 2:
        raise ValueError(f"Unsupported array shape {X.shape}; expected 2D or 3D with 3×3")
    for name in ('row', 'col', 'rowT', 'colT'):
        try:
            out[name] = _apply_map_2d(X, name).astype(np.float64, copy=False)
        except Exception:
            pass
    if not out:
        raise ValueError("Could not form any 3×3 candidates from input array.")
    return out


# -----------------------------------------------------------------------------
# Physics scoring to pick the right permutation
# -----------------------------------------------------------------------------

def score_A_incompressible(A: np.ndarray) -> float:
    tr = np.trace(A, axis1=1, axis2=2)
    fro = np.sqrt(np.sum(A * A, axis=(1, 2)) + 1e-30)
    return -float(np.mean(np.abs(tr) / (fro + 1e-30)))  # closer to 0 is better


def score_P_symmetric(P: np.ndarray) -> float:
    sym = 0.5 * (P + np.swapaxes(P, -1, -2))
    skew = P - sym
    fro = np.sqrt(np.sum(P * P, axis=(1, 2)) + 1e-30)
    fro_skew = np.sqrt(np.sum(skew * skew, axis=(1, 2)) + 1e-30)
    return -float(np.mean(fro_skew / (fro + 1e-30)))  # smaller skew is better


def pick_mapping_A(cands: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray, float]:
    best = max(((score_A_incompressible(T), k) for k, T in cands.items()), key=lambda x: x[0])
    name = best[1]
    return name, cands[name], best[0]


def pick_mapping_P(cands: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray, float]:
    best = max(((score_P_symmetric(T), k) for k, T in cands.items()), key=lambda x: x[0])
    name = best[1]
    return name, cands[name], best[0]


# -----------------------------------------------------------------------------
# Derived DNS quantities
# -----------------------------------------------------------------------------

def derive_dns(A: np.ndarray, P: np.ndarray) -> Dict[str, np.ndarray]:
    # Magnitude and normalized gradient
    eps = np.sqrt(np.sum(A * A, axis=(1, 2)) + 1e-30)
    a = A / eps[:, None, None]
    s = 0.5 * (a + np.swapaxes(a, -1, -2))
    w = 0.5 * (a - np.swapaxes(a, -1, -2))

    # Invariants of a
    a2 = np.einsum('bij,bjk->bik', a, a, optimize=True)
    q = -0.5 * np.trace(a2, axis1=1, axis2=2)
    a3 = np.einsum('bij,bjk->bik', a2, a, optimize=True)
    r = -1.0 / 3.0 * np.trace(a3, axis1=1, axis2=2)

    # Anisotropic Q from P
    trP = np.trace(P, axis1=1, axis2=2)[:, None, None]
    I = np.eye(3)[None, :, :]
    Q = P - (trP / 3.0) * I

    # Q prime and normalized
    eps2 = (eps ** 2)[:, None, None]
    Qp = Q / (eps2 + 1e-30)
    psi = np.sqrt(np.sum(Qp * Qp, axis=(1, 2)) + 1e-30)
    Qhat = Qp / psi[:, None, None]
    # Enforce symmetry of Qhat numerically
    Qhat = 0.5 * (Qhat + np.swapaxes(Qhat, -1, -2))

    return dict(A=A, P=P, s=s, w=w, eps=eps, q=q, r=r, Q=Q, Qp=Qp, psi=psi, Qhat=Qhat)


# -----------------------------------------------------------------------------
# Plot utilities
# -----------------------------------------------------------------------------

def pdf_curve(x: np.ndarray, bins: int = 140, clip: Tuple[float, float] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).reshape(-1)
    if clip is not None:
        x = x[(x >= clip[0]) & (x <= clip[1])]
    if x.size == 0:
        xc = np.linspace(0, 1, bins)
        return xc, np.zeros_like(xc)
    h, edges = np.histogram(x, bins=bins, density=True)
    xc = 0.5 * (edges[:-1] + edges[1:])
    return xc, h


def unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + 1e-30)


def vorticity_from_w(w: np.ndarray) -> np.ndarray:
    wx = w[:, 2, 1] - w[:, 1, 2]
    wy = w[:, 0, 2] - w[:, 2, 0]
    wz = w[:, 1, 0] - w[:, 0, 1]
    return unit_rows(np.stack([wx, wy, wz], axis=1))


def eigh_asc(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(M)
    return vals, vecs  # ascending: (γ, β, α)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def fig9_align_s_vs_Qhat(data: Dict[str, np.ndarray], outdir: Path) -> None:
    s = data['s']
    Qhat = data['Qhat']
    _, Vs = eigh_asc(s)
    _, Vq = eigh_asc(Qhat)
    s_e = [Vs[:, :, 0], Vs[:, :, 1], Vs[:, :, 2]]
    q_e = [Vq[:, :, 0], Vq[:, :, 1], Vq[:, :, 2]]

    def cosabs(U, V):
        U = unit_rows(U); V = unit_rows(V)
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

    fig, axs = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for i in range(3):
        for j in range(3):
            xs, ys = pdf_curve(cosabs(s_e[i], q_e[j]), bins=140, clip=(0, 1))
            ax = axs[i, j]
            ax.plot(xs, ys, 'k-', lw=2)
            ax.set_xlim(0, 1)
            ax.set_xlabel('|cos θ|')
            ax.set_ylabel('PDF')
            ax.set_title(labels[i][j])
            ax.grid(True, ls=':', alpha=0.5)
    fig.suptitle('Fig. 9 — Alignment: S vs Q̂′ (DNS)', y=0.995)
    fig.savefig(outdir / 'fig9_align_s_vs_Qhat0.png', dpi=600)
    plt.close(fig)


def fig10_align_w_vs_Qhat(data: Dict[str, np.ndarray], outdir: Path) -> None:
    w = data['w']
    Qhat = data['Qhat']
    omg = vorticity_from_w(w)
    _, Vq = eigh_asc(Qhat)
    q_e = [Vq[:, :, 0], Vq[:, :, 1], Vq[:, :, 2]]

    def cosabs(u, V):
        u = unit_rows(u); V = unit_rows(V)
        return np.clip(np.abs(np.sum(u * V, axis=1)), 0.0, 1.0)

    labels = [r'$\hat e_{\gamma_p}\!\cdot\!\hat\omega$',
              r'$\hat e_{\beta_p}\!\cdot\!\hat\omega$',
              r'$\hat e_{\alpha_p}\!\cdot\!\hat\omega$']

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for j in range(3):
        xs, ys = pdf_curve(cosabs(omg, q_e[j]), bins=140, clip=(0, 1))
        ax = axs[j]
        ax.plot(xs, ys, 'k-', lw=2)
        ax.set_xlim(0, 1)
        ax.set_xlabel('|cos θ|')
        ax.set_ylabel('PDF')
        ax.set_title(labels[j])
        ax.grid(True, ls=':', alpha=0.5)
    fig.suptitle('Fig. 10 — Alignment: ω vs Q̂′ (DNS)')
    fig.savefig(outdir / 'fig10_align_w_vs_Qhat0.png', dpi=600)
    plt.close(fig)


def fig11_pdf_psi(data: Dict[str, np.ndarray], outdir: Path) -> None:
    psi = data['psi']
    xs, ys = pdf_curve(psi, bins=160, clip=(np.percentile(psi, 0.1), np.percentile(psi, 99.9)))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(xs, ys, 'k-', lw=2)
    ax.set_xlabel(r'$\psi$')
    ax.set_ylabel('PDF')
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_title('Fig. 11 — PDF of $\psi$ (DNS)')
    fig.savefig(outdir / 'fig11_pdf_psi.png', dpi=600)
    plt.close(fig)


def _binned_mean(x: np.ndarray, y: np.ndarray, lo: float, hi: float, nbins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    sums = np.bincount(idx, weights=y, minlength=nbins).astype(np.float64)
    counts = np.bincount(idx, minlength=nbins).astype(np.int64)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / np.maximum(counts, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means, counts


def fig12_phi_vs_Qsigma(data: Dict[str, np.ndarray], outdir: Path) -> None:
    eps = data['eps']
    q = data['q']
    Q = data['Q']

    # x = (q·ε²)/σ(q·ε²)
    Qraw = q * (eps ** 2)
    sig_x = float(np.std(Qraw, ddof=1) + 1e-30)
    x = Qraw / sig_x

    # y = ⟨QmnQmn | x-bin⟩ · σ(x)/σ(QmnQmn) with σ(x)=1 by construction
    Qmag2 = np.sum(Q * Q, axis=(1, 2))
    sig_y = float(np.std(Qmag2, ddof=1) + 1e-30)

    # Full range domain (robust)
    lo_full = float(np.percentile(x, 0.5)); hi_full = float(np.percentile(x, 99.5))
    cx_f, mu_y_f_raw, cnt_f = _binned_mean(x, Qmag2, lo_full, hi_full, nbins=140)
    mu_y_f = mu_y_f_raw * (sig_x / sig_y)

    # Zoom domain [-1,1]
    cx_z, mu_y_z_raw, cnt_z = _binned_mean(x, Qmag2, -1.0, 1.0, nbins=80)
    mu_y_z = mu_y_z_raw * (sig_x / sig_y)

    # Plot full
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    m = cnt_f >= 1
    ax.plot(cx_f[m], mu_y_f[m], 'k-', lw=2.0)
    ax.set_xlabel(r'$Q/\sigma_Q$')
    ax.set_ylabel(r'$\langle Q_{mn}Q_{mn}\mid Q/\sigma_Q\rangle\,\,\sigma_Q/\sigma_{Q_{mn}Q_{mn}}$')
    ax.set_title('Fig. 12 — $\phi$ vs $Q/\sigma_Q$ (DNS, full)')
    ax.grid(True, ls=':', alpha=0.55)
    fig.savefig(outdir / 'fig12_phi_vs_Qsigma_full.png', dpi=600, bbox_inches='tight')
    plt.close(fig)

    # Plot zoom
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    m = cnt_z >= 1
    ax.plot(cx_z[m], mu_y_z[m], 'k-', lw=2.0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 0.25)
    ax.set_xlabel(r'$Q/\sigma_Q$')
    ax.set_ylabel(r'$\langle Q_{mn}Q_{mn}\mid Q/\sigma_Q\rangle\,\,\sigma_Q/\sigma_{Q_{mn}Q_{mn}}$')
    ax.set_title('Fig. 12 — $\phi$ vs $Q/\sigma_Q$ (DNS, zoom)')
    ax.grid(True, ls=':', alpha=0.55)
    fig.savefig(outdir / 'fig12_phi_vs_Qsigma_zoom.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(description='DNS-only Figures 9–12 (from scratch)')
    pa.add_argument('--vel', default='PANN_PressureHessianTensor/velGrad.mat')
    pa.add_argument('--ph', default='PANN_PressureHessianTensor/PH.mat')
    pa.add_argument('--vel-var', default=None, help='Variable name in velGrad.mat (e.g., velGrad)')
    pa.add_argument('--ph-var', default=None, help='Variable name in PH.mat (e.g., PH)')
    pa.add_argument('--mapA', default=None, choices=[None, 'row', 'col', 'rowT', 'colT'], help='Force A mapping')
    pa.add_argument('--mapP', default=None, choices=[None, 'row', 'col', 'rowT', 'colT'], help='Force P mapping')
    pa.add_argument('--max-samples', type=int, default=300000)
    pa.add_argument('--out', default='PANN_PressureHessianTensor/figs')
    args = pa.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Load raw arrays
    vel_names = (args.vel_var,) if args.vel_var else ("velGrad", "A", "velgrad")
    ph_names = (args.ph_var,) if args.ph_var else ("PH", "Q", "P")
    vg_raw = load_mat_array(args.vel, vel_names)
    ph_raw = load_mat_array(args.ph, ph_names)

    # Build candidates for A and P; force mapping if requested
    A_cands = reconstruct_3x3(vg_raw)
    P_cands = reconstruct_3x3(ph_raw)
    if args.mapA and args.mapA in A_cands:
        A_map, A_t = args.mapA, A_cands[args.mapA]
        A_score = score_A_incompressible(A_t)
    else:
        A_map, A_t, A_score = pick_mapping_A(A_cands)

    if args.mapP and args.mapP in P_cands:
        P_map, P_t = args.mapP, P_cands[args.mapP]
        P_score = score_P_symmetric(P_t)
    else:
        P_map, P_t, P_score = pick_mapping_P(P_cands)

    # Subsample to limit eigen computations if requested
    N = A_t.shape[0]
    if args.max_samples and N > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=args.max_samples, replace=False)
        A_t = A_t[idx]
        P_t = P_t[idx]

    # Derive DNS quantities
    data = derive_dns(A_t.astype(np.float64), P_t.astype(np.float64))

    # Write a small mapping report
    try:
        with open(outdir / 'mapping_report.txt', 'w') as f:
            print({'A_map': A_map, 'A_score': A_score, 'P_map': P_map, 'P_score': P_score}, file=f)
            trQ = float(np.mean(np.abs(np.trace(data['Q'], axis1=1, axis2=2))))
            froQ = float(np.mean(np.sqrt(np.sum(data['Q']*data['Q'], axis=(1,2))+1e-30)))
            print({'mean_abs_trQ': trQ, 'mean_froQ': froQ}, file=f)
    except Exception:
        pass

    # Plots
    fig9_align_s_vs_Qhat(data, outdir)
    fig10_align_w_vs_Qhat(data, outdir)
    fig11_pdf_psi(data, outdir)
    fig12_phi_vs_Qsigma(data, outdir)

    print(f"Saved figures to {outdir} (A_map={A_map}, P_map={P_map})")


if __name__ == '__main__':
    main()

