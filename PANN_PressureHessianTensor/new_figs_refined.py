#!/usr/bin/env python3
import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

EPS = 1e-14


def load_mat(path):
    mat = scipy.io.loadmat(path)
    key = [k for k in mat if not k.startswith('__')][0]
    data = np.asarray(mat[key], dtype=np.float64)
    if data.shape[0] == 9:
        data = data.T
    assert data.shape[1] == 9, f"Expected Nx9, got {data.shape}"
    return data


def reshape_tensors(V, mapping='matlab_col'):
    N = V.shape[0]
    T = np.zeros((N, 3, 3), dtype=np.float64)
    if mapping == 'matlab_col':
        T[:, :, 0] = V[:, 0:3]
        T[:, :, 1] = V[:, 3:6]
        T[:, :, 2] = V[:, 6:9]
    else:
        T[:, 0, :] = V[:, 0:3]
        T[:, 1, :] = V[:, 3:6]
        T[:, 2, :] = V[:, 6:9]
    return T


def detect_mapping(V_A, V_Q, sample_size=50000):
    N = min(V_A.shape[0], sample_size)
    idx = np.random.choice(V_A.shape[0], N, replace=False)
    scores = {}
    for mapping in ['matlab_col', 'row_major']:
        A = reshape_tensors(V_A[idx], mapping)
        Q = reshape_tensors(V_Q[idx], mapping)
        tr_A = np.abs(A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2])
        tr_Q = np.abs(Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2])
        score = np.median(tr_A) + np.median(tr_Q)
        scores[mapping] = score
        print(f"{mapping}: median|Tr[A]|={np.median(tr_A):.2e}, median|Tr[Q]|={np.median(tr_Q):.2e}")
    best = min(scores, key=scores.get)
    print(f"-> Selected mapping: {best}\n")
    return best


def construct_anisotropic_Q(H_np):
    trace = H_np[:, 0, 0] + H_np[:, 1, 1] + H_np[:, 2, 2]
    return H_np - (trace[:, None, None] / 3.0) * np.eye(3)[None, :, :]


def frobenius_norm(tensors):
    return np.linalg.norm(tensors.reshape(tensors.shape[0], -1), axis=1)


def gaussian_kernel(sigma):
    if sigma is None or sigma <= 0:
        return None
    radius = int(max(1, round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_array(values, kernel):
    if kernel is None:
        return values
    padded = np.pad(values, (kernel.size // 2,), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


def histogram_pdf(values, bins, scale=1.0, value_range=None, smooth_kernel=None):
    values = np.asarray(values)
    if values.size == 0:
        if value_range is None:
            value_range = (0.0, 1.0)
        edges = np.linspace(value_range[0], value_range[1], bins + 1)
        centers = edges[:-1] + 0.5 * np.diff(edges)
        return centers, np.zeros_like(centers)
    if value_range is None:
        value_range = (float(np.min(values)), float(np.max(values)))
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    widths = np.diff(edges)
    total = counts.sum()
    if total == 0:
        centers = edges[:-1] + widths / 2.0
        return centers, np.zeros_like(centers)
    pdf = counts / (total * widths)
    pdf *= scale
    if smooth_kernel is not None:
        pdf = smooth_array(pdf, smooth_kernel)
    centers = edges[:-1] + widths / 2.0
    return centers, pdf


def describe_metric(label, values, ideal=0.0):
    values = np.asarray(values)
    abs_vals = np.abs(values)
    return {
        'label': label,
        'ideal': ideal,
        'mean': values.mean(),
        'mean_abs': abs_vals.mean(),
        'median_abs': np.median(abs_vals),
        'p95_abs': np.percentile(abs_vals, 95),
        'p99_abs': np.percentile(abs_vals, 99),
        'max_abs': abs_vals.max(),
    }


def print_metric(metric):
    print(
        f"{metric['label']}: mean={metric['mean']:.3e}, "
        f"mean|.|={metric['mean_abs']:.3e}, median|.|={metric['median_abs']:.3e}, "
        f"p95|.|={metric['p95_abs']:.3e}, p99|.|={metric['p99_abs']:.3e}, max|.|={metric['max_abs']:.3e}"
    )
    if metric['ideal'] is not None:
        print(f"  ideal -> {metric['ideal']:.3e}")


def compute_diagnostics(A_np, H_np, Q_np):
    diag = {}
    tr_A = A_np[:, 0, 0] + A_np[:, 1, 1] + A_np[:, 2, 2]
    tr_H = H_np[:, 0, 0] + H_np[:, 1, 1] + H_np[:, 2, 2]
    tr_Q = Q_np[:, 0, 0] + Q_np[:, 1, 1] + Q_np[:, 2, 2]
    fro_A = frobenius_norm(A_np)
    fro_Q = frobenius_norm(Q_np)
    A2_trace = np.einsum('nij,nji->n', A_np, A_np)
    mismatch = tr_H + A2_trace
    rel_div = np.abs(tr_A) / np.maximum(fro_A, EPS)
    diag['metrics'] = [
        describe_metric('Tr[A]', tr_A, ideal=0.0),
        describe_metric('Tr[H]', tr_H, ideal=0.0),
        describe_metric('Tr[Q]', tr_Q, ideal=0.0),
        describe_metric('||A||_F', fro_A, ideal=None),
        describe_metric('||Q||_F', fro_Q, ideal=None),
        describe_metric('Tr[H] + Tr(A^2)', mismatch, ideal=0.0),
        describe_metric('Tr[A] / ||A||_F', rel_div, ideal=0.0),
    ]
    diag['corr_trH_vs_trA2'] = np.corrcoef(tr_H, -A2_trace)[0, 1]
    diag['samples'] = A_np.shape[0]
    return diag, tr_A, tr_H, tr_Q, A2_trace, mismatch, fro_A


def report_diagnostics(diag):
    print(f"Total samples: {diag['samples']}")
    for metric in diag['metrics']:
        print_metric(metric)
    print(f"Correlation Tr[H] vs -Tr(A^2): {diag['corr_trH_vs_trA2']:.5f}\n")


def apply_filters(tr_A, mismatch, fro_A, mask, args):
    reasons = []
    if args.max_abs_div is not None:
        keep = np.abs(tr_A) <= args.max_abs_div
        mask &= keep
        reasons.append(('|Tr[A]|', keep))
    if args.max_rel_div is not None:
        keep = np.abs(tr_A) / np.maximum(fro_A, EPS) <= args.max_rel_div
        mask &= keep
        reasons.append(('|Tr[A]|/||A||_F', keep))
    if args.max_trace_mismatch is not None:
        keep = np.abs(mismatch) <= args.max_trace_mismatch
        mask &= keep
        reasons.append(('|Tr[H]+Tr(A^2)|', keep))
    return mask, reasons


def summarize_filters(mask, reasons):
    total = mask.size
    kept = mask.sum()
    print(f"Filter pass rate: {kept}/{total} ({100 * kept / max(total, 1):.2f}%)")
    for label, keep in reasons:
        removed = (~keep).sum()
        if removed:
            print(f"  Removed {removed} samples by {label}")
    print()


def compute_normalized_tensors(A_np, Q_np):
    A_t = torch.from_numpy(A_np).to(torch.float64)
    Q_t = torch.from_numpy(Q_np).to(torch.float64)
    e2 = torch.sum(A_t * A_t, dim=(-2, -1))
    e = torch.sqrt(e2.clamp(min=EPS))
    a = A_t / e.view(-1, 1, 1)
    s = 0.5 * (a + a.transpose(-1, -2))
    w = 0.5 * (a - a.transpose(-1, -2))
    Q0 = Q_t / e2.view(-1, 1, 1)
    omega = torch.sqrt(torch.sum(Q0 * Q0, dim=(-2, -1)).clamp(min=EPS))
    Qhat0 = Q0 / omega.view(-1, 1, 1)
    return {'a': a, 's': s, 'w': w, 'Q0': Q0, 'Qhat0': Qhat0, 'omega': omega}


def compute_eigensystem(M_torch):
    evals, evecs = torch.linalg.eigh(M_torch)
    idx = torch.argsort(evals, dim=-1, descending=True)
    evals_sorted = torch.gather(evals, -1, idx)
    idx_exp = idx.unsqueeze(-2).expand(-1, 3, -1)
    evecs_sorted = torch.gather(evecs, -1, idx_exp)
    return evals_sorted, evecs_sorted


def compute_vorticity(w_torch):
    vort = torch.stack([
        w_torch[:, 2, 1],
        w_torch[:, 0, 2],
        w_torch[:, 1, 0],
    ], dim=1)
    return torch.nn.functional.normalize(vort, dim=-1, eps=EPS)


def fig9_alignment_s_Qhat0(A_np, Q_np, bins=80, smooth_sigma=1.0):
    tens = compute_normalized_tensors(A_np, Q_np)
    _, E_s = compute_eigensystem(tens['s'])
    _, E_Q = compute_eigensystem(tens['Qhat0'])
    E_s = torch.nn.functional.normalize(E_s, dim=-2, eps=EPS)
    E_Q = torch.nn.functional.normalize(E_Q, dim=-2, eps=EPS)
    cos_mat = torch.abs(torch.einsum('nai,naj->nij', E_s, E_Q)).cpu().numpy()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    labels_s = ['e_a^s', 'e_b^s', 'e_c^s']
    labels_Q = ['e_a^p', 'e_b^p', 'e_c^p']
    kernel = gaussian_kernel(smooth_sigma)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            centers, pdf = histogram_pdf(
                cos_mat[:, i, j],
                bins=bins,
                scale=0.5,
                smooth_kernel=kernel
            )
            ax.plot(centers, pdf, color='C0', lw=1.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2)
            ax.set_xlabel('|cos(theta)|')
            ax.set_ylabel('PDF')
            ax.set_title(f'{labels_s[i]} vs {labels_Q[j]}')
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig10_alignment_vorticity_Qhat0(A_np, Q_np, bins=80, smooth_sigma=1.0):
    tens = compute_normalized_tensors(A_np, Q_np)
    vort = compute_vorticity(tens['w'])
    _, E_Q = compute_eigensystem(tens['Qhat0'])
    E_Q = torch.nn.functional.normalize(E_Q, dim=-2, eps=EPS)
    cos_vals = torch.abs(torch.einsum('ni,nij->nj', vort, E_Q)).cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ['e_a^p', 'e_b^p', 'e_c^p']
    kernel = gaussian_kernel(smooth_sigma)
    for j, ax in enumerate(axes):
        centers, pdf = histogram_pdf(
            cos_vals[:, j],
            bins=bins,
            scale=0.5,
            smooth_kernel=kernel
        )
        ax.plot(centers, pdf, color='C0', lw=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        ax.set_xlabel('|cos(theta)|')
        ax.set_ylabel('PDF')
        ax.set_title(f'omega_hat vs {labels[j]}')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig11_pdf_omega(A_np, Q_np, bins=180, clip_percentile=99.5, smooth_sigma=1.2):
    tens = compute_normalized_tensors(A_np, Q_np)
    omega = tens['omega'].cpu().numpy()
    omega_max = np.percentile(omega, clip_percentile)
    omega_clipped = omega[(omega >= 0.0) & (omega <= omega_max)]
    kernel = gaussian_kernel(smooth_sigma)
    centers, pdf = histogram_pdf(
        omega_clipped,
        bins=bins,
        value_range=(0.0, float(omega_max)),
        smooth_kernel=kernel
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, pdf, color='C0', lw=1.8)
    ax.set_xlim(0.0, float(omega_max))
    ax.set_xlabel('omega = ||Q^0||_F')
    ax.set_ylabel('PDF')
    ax.set_title('Fig. 11: PDF of omega')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig12_phi_vs_qe2(
    A_np,
    Q_np,
    bins=120,
    x_limit=1.0,
    y_limit=0.25,
    min_count=40,
    smooth_sigma=1.2
):
    A_t = torch.from_numpy(A_np).to(torch.float64)
    Q_t = torch.from_numpy(Q_np).to(torch.float64)

    # x-axis: Q = q e^2 = -0.5 Tr(A^2)
    A_sq = torch.matmul(A_t, A_t)
    qe2 = -0.5 * torch.einsum('nii->n', A_sq)

    # y-axis: ||Q||_F^2
    Q2 = torch.sum(Q_t * Q_t, dim=(-2, -1))

    qe2_np = qe2.cpu().numpy()
    Q2_np = Q2.cpu().numpy()

    sigma_x = np.std(qe2_np, ddof=1)
    sigma_y = np.std(Q2_np, ddof=1)

    x_norm = qe2_np / (sigma_x + EPS)
    y_norm = Q2_np / (sigma_y + EPS)

    mask = (x_norm >= -x_limit) & (x_norm <= x_limit)
    x_sel = x_norm[mask]
    y_sel = y_norm[mask]

    edges = np.linspace(-x_limit, x_limit, bins + 1)
    idx = np.clip(np.digitize(x_sel, edges) - 1, 0, bins - 1)
    sums = np.bincount(idx, weights=y_sel, minlength=bins).astype(np.float64)
    counts = np.bincount(idx, minlength=bins).astype(np.int64)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / np.maximum(counts, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    valid = counts >= min_count
    kernel = gaussian_kernel(smooth_sigma)
    means_valid = means.copy()
    means_valid[~valid] = np.nan
    nan_mask = ~np.isnan(means_valid)
    if kernel is not None:
        # replace NaNs with zero for convolution, then mask afterwards
        temp = means_valid.copy()
        temp[~nan_mask] = 0.0
        smoothed = smooth_array(temp, kernel)
        weight = smooth_array(nan_mask.astype(np.float64), kernel)
        with np.errstate(invalid='ignore', divide='ignore'):
            smoothed = smoothed / np.maximum(weight, 1e-12)
        means_valid = smoothed

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers[valid], means_valid[valid], 'C0-', lw=2.0)
    ax.set_xlabel('Q / sigma_Q')
    ax.set_ylabel('phi(Q / sigma_Q)')
    ax.set_title('Fig. 12: phi vs Q/sigma_Q')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0.0, y_limit)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='DNS diagnostics and figure generation with physical filters')
    parser.add_argument('--velgrad', required=True, help='Path to velGrad.mat')
    parser.add_argument('--ph', required=True, help='Path to PH.mat containing full pressure Hessian')
    parser.add_argument('--mapping', default='auto', choices=['auto', 'matlab_col', 'row_major'])
    parser.add_argument('--out', default='./figs_dns_refined', help='Output directory for figures')
    parser.add_argument('--bins', type=int, default=120, help='Histogram bins for PDF plots')
    parser.add_argument('--max-abs-div', type=float, default=None, help='Reject samples with |Tr[A]| above this threshold')
    parser.add_argument('--max-rel-div', type=float, default=None, help='Reject samples with |Tr[A]|/||A||_F above this threshold')
    parser.add_argument('--max-trace-mismatch', type=float, default=None, help='Reject samples with |Tr[H] + Tr(A^2)| above this threshold')
    parser.add_argument('--max-samples', type=int, default=None, help='Randomly subsample this many points after filtering')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for subsampling')
    parser.add_argument('--dry-run', action='store_true', help='Only print diagnostics, skip figure generation')
    parser.add_argument('--phi-x-limit', type=float, default=1.0, help='Half-width of qe^2 axis for Fig. 12')
    parser.add_argument('--phi-y-limit', type=float, default=0.25, help='Upper limit of phi axis for Fig. 12')
    parser.add_argument('--smooth-sigma', type=float, default=1.0, help='Gaussian smoothing (in bins) for alignment PDFs')
    parser.add_argument('--omega-bins', type=int, default=220, help='Histogram bins for Fig. 11')
    parser.add_argument('--omega-smooth-sigma', type=float, default=1.2, help='Gaussian smoothing (in bins) for Fig. 11 PDF')
    parser.add_argument('--phi-bins', type=int, default=140, help='Number of bins for Fig. 12')
    parser.add_argument('--phi-min-count', type=int, default=40, help='Minimum samples per bin for Fig. 12')
    parser.add_argument('--phi-smooth-sigma', type=float, default=1.2, help='Gaussian smoothing (in bins) for Fig. 12 curve')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    V_A = load_mat(args.velgrad)
    V_Q = load_mat(args.ph)
    assert V_A.shape == V_Q.shape, f'Shape mismatch: {V_A.shape} vs {V_Q.shape}'

    if args.mapping == 'auto':
        mapping = detect_mapping(V_A, V_Q)
    else:
        mapping = args.mapping
        print(f'Using mapping: {mapping}\n')

    A_np = reshape_tensors(V_A, mapping)
    H_np = reshape_tensors(V_Q, mapping)
    Q_np = construct_anisotropic_Q(H_np)

    diag, tr_A, tr_H, tr_Q, A2_trace, mismatch, fro_A = compute_diagnostics(A_np, H_np, Q_np)
    print('Raw diagnostics (before filtering):')
    report_diagnostics(diag)

    mask = np.ones(A_np.shape[0], dtype=bool)
    mask, reasons = apply_filters(tr_A, mismatch, fro_A, mask, args)
    summarize_filters(mask, reasons)

    A_np_f = A_np[mask]
    Q_np_f = Q_np[mask]
    H_np_f = H_np[mask]

    if args.max_samples is not None and mask.sum() > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(mask.sum(), size=args.max_samples, replace=False)
        A_np_f = A_np_f[idx]
        Q_np_f = Q_np_f[idx]
        H_np_f = H_np_f[idx]
        print(f'Subsampled to {args.max_samples} samples\n')

    diag_f, *_ = compute_diagnostics(A_np_f, H_np_f, Q_np_f)
    print('Diagnostics after filtering:')
    report_diagnostics(diag_f)

    if args.dry_run:
        print('Dry run; skipping figure generation.')
        return

    print('Generating Fig. 9...')
    fig9 = fig9_alignment_s_Qhat0(A_np_f, Q_np_f, bins=args.bins, smooth_sigma=args.smooth_sigma)
    fig9.savefig(out_dir / 'fig9_dns.png', dpi=200)
    plt.close(fig9)

    print('Generating Fig. 10...')
    fig10 = fig10_alignment_vorticity_Qhat0(A_np_f, Q_np_f, bins=args.bins, smooth_sigma=args.smooth_sigma)
    fig10.savefig(out_dir / 'fig10_dns.png', dpi=200)
    plt.close(fig10)

    print('Generating Fig. 11...')
    fig11 = fig11_pdf_omega(
        A_np_f,
        Q_np_f,
        bins=args.omega_bins,
        smooth_sigma=args.omega_smooth_sigma
    )
    fig11.savefig(out_dir / 'fig11_dns.png', dpi=200)
    plt.close(fig11)

    print('Generating Fig. 12...')
    fig12 = fig12_phi_vs_qe2(
        A_np_f,
        Q_np_f,
        bins=args.phi_bins,
        x_limit=args.phi_x_limit,
        y_limit=args.phi_y_limit,
        min_count=args.phi_min_count,
        smooth_sigma=args.phi_smooth_sigma
    )
    fig12.savefig(out_dir / 'fig12_dns.png', dpi=200)
    plt.close(fig12)

    print(f"\n✓ Saved plots to {out_dir.resolve()}")


if __name__ == '__main__':
    main()
