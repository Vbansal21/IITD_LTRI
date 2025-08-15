import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_psi_contour_on_qr(q, r, psi, savepath=None):
    plt.figure(figsize=(6,5))
    hb = plt.hexbin(r, q, C=psi, gridsize=40, cmap='magma', reduce_C_function=np.mean)
    plt.colorbar(hb, label='⟨ψ | q,r⟩')
    plt.xlabel('r')
    plt.ylabel('q')
    plt.title('Conditional mean ψ')
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_qr_joint_pdf(q, r, savepath=None):
    plt.figure(figsize=(6,5))
    sns.kdeplot(x=r, y=q, cmap='Blues', fill=True, thresh=0.02)
    plt.xlabel('r')
    plt.ylabel('q')
    plt.title('Joint PDF (q,r)')
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_invariant_ratios(A, Q_hat_prime_pred, model_Q, get_tensor_derivatives, device, savepath=None):
    # Apply 30 degree rotation about z-axis
    θ = np.pi / 6
    R = torch.tensor([[np.cos(θ), -np.sin(θ), 0],
                      [np.sin(θ),  np.cos(θ), 0],
                      [0,           0,        1]], dtype=torch.float32, device=device)
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    A_rot = torch.einsum('ij,bjk,lk->bil', R, A, R)
    with torch.no_grad():
        s_rot, w_rot, _, _, _ = get_tensor_derivatives(A_rot)
        Q_rot = model_Q(s_rot, w_rot).cpu().numpy()
    # eigenvalues
    λ      = np.linalg.eigvalsh(Q_hat_prime_pred)
    λ_rot  = np.linalg.eigvalsh(Q_rot)
    ratios = λ_rot / (λ + 1e-9)
    plt.figure(figsize=(6,4))
    for i, ls in zip(range(3), ['-', '--', ':']):
        sns.kdeplot(ratios[:, i], label=f'F{i+1}', linestyle=ls)
    plt.title('Invariant ratios F₁–F₃'); plt.xlim(0,2); plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_eigenvector_alignment_s_Q(s, Q_hat_prime, Q_hat_prime_pred, savepath=None):
    def evec(X):
        return np.linalg.eigh(X)[1]
    e_s  = evec(s)
    e_Q  = evec(Q_hat_prime)
    e_Qp = evec(Q_hat_prime_pred)
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
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_eigenvector_alignment_vorticity_Q(w, Q_hat_prime, Q_hat_prime_pred, savepath=None):
    def evec(X):
        return np.linalg.eigh(X)[1]
    ω = np.stack([w[:,2,1]-w[:,1,2],
                  w[:,0,2]-w[:,2,0],
                  w[:,1,0]-w[:,0,1]], axis=1)
    ω /= (np.linalg.norm(ω,axis=1,keepdims=True)+1e-9)
    e_Q  = evec(Q_hat_prime)
    e_Qp = evec(Q_hat_prime_pred)
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
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_psi_pdf(psi_true, psi_pred, savepath=None):
    plt.figure(figsize=(6,4))
    sns.kdeplot(psi_true,label='DNS')
    sns.kdeplot(psi_pred,label='Pred',ls='--')
    plt.title('PDF of ψ')
    plt.xlim(left=0)
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_phi_vs_q_epsilon_sq(Q_prime, Q_hat_prime_pred, psi_pred, q, epsilon, savepath=None):
    φ_true = (Q_prime**2).sum((1,2))
    φ_pred = (Q_hat_prime_pred*(psi_pred[:,None,None]))**2
    φ_pred = φ_pred.sum((1,2))
    qε2    = q * (epsilon**2)
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
    if savepath: plt.savefig(savepath)
    plt.show()

# --- Also include all the previous advanced scientific plots ---
def plot_reliability_curve(bin_uncertainties, bin_errors, label=None, savepath=None):
    plt.figure(figsize=(5, 5))
    plt.plot(bin_uncertainties, bin_errors, 'o-', lw=2, label=label or "Model")
    plt.plot([min(bin_uncertainties), max(bin_uncertainties)],
             [min(bin_uncertainties), max(bin_uncertainties)], 'r--', label='Perfect Calibration')
    plt.xlabel('Predicted Uncertainty')
    plt.ylabel('Observed Error')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_trace_conservation(Q_true, Q_pred, savepath=None):
    tr_true = np.trace(Q_true, axis1=1, axis2=2)
    tr_pred = np.trace(Q_pred, axis1=1, axis2=2)
    plt.figure(figsize=(6,4))
    sns.kdeplot(tr_true, label='True', lw=2)
    sns.kdeplot(tr_pred, label='Pred', lw=2, ls='--')
    plt.xlabel('Tr(Q)')
    plt.title('Tracelessness of Q′')
    plt.legend()
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_eigenvalue_distribution(Q_true, Q_pred, savepath=None):
    eigs_true = np.linalg.eigvalsh(Q_true)
    eigs_pred = np.linalg.eigvalsh(Q_pred)
    plt.figure(figsize=(7, 4))
    for i in range(3):
        sns.kdeplot(eigs_true[:, i], label=f'True λ{i+1}', lw=2)
        sns.kdeplot(eigs_pred[:, i], label=f'Pred λ{i+1}', lw=2, ls='--')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title('Q′ Eigenvalue Distribution')
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_spectral_psd(errors, savepath=None):
    errors_flat = errors.reshape(errors.shape[0], -1)
    psd_avg = np.zeros(errors_flat.shape[1])
    for i in range(min(100, errors_flat.shape[0])):
        fft_vals = np.fft.fft(errors_flat[i])
        psd = np.abs(fft_vals) ** 2
        psd_avg += psd
    psd_avg /= min(100, errors_flat.shape[0])
    freqs = np.fft.fftfreq(errors_flat.shape[1])
    plt.figure(figsize=(7, 4))
    plt.loglog(freqs[freqs > 0], psd_avg[freqs > 0])
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectral Density of Errors')
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_energy_scale_consistency(epsilon, Q_prime, savepath=None):
    Q_energy = np.sum(Q_prime ** 2, axis=(1,2))
    eps2 = epsilon ** 2
    plt.figure(figsize=(6,4))
    plt.scatter(eps2, Q_energy, alpha=0.5, s=8)
    plt.xlabel('ε²')
    plt.ylabel('||Q′||²')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Energy Scale Consistency: ||Q′||² vs ε²')
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

# ---- A single call for all paper/advanced plots, for convenience ----
def all_advanced_plots(
    q, r, psi_true, psi_pred,
    A, s, w,
    Q_prime, Q_hat_prime, Q_hat_prime_pred,
    model_Q, get_tensor_derivatives, device,
    epsilon,  # vector
    save_dir='figs'
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all advanced plots to {save_dir}")

    plot_psi_contour_on_qr(q, r, psi_pred, savepath=os.path.join(save_dir,'psi_contour_on_qr.png'))
    plot_qr_joint_pdf(q, r, savepath=os.path.join(save_dir,'qr_joint_pdf.png'))
    plot_invariant_ratios(A, Q_hat_prime_pred, model_Q, get_tensor_derivatives, device, savepath=os.path.join(save_dir,'invariant_ratios.png'))
    plot_eigenvector_alignment_s_Q(s, Q_hat_prime, Q_hat_prime_pred, savepath=os.path.join(save_dir,'eigenvector_alignment_s_Q.png'))
    plot_eigenvector_alignment_vorticity_Q(w, Q_hat_prime, Q_hat_prime_pred, savepath=os.path.join(save_dir,'eigenvector_alignment_vorticity_Q.png'))
    plot_psi_pdf(psi_true, psi_pred, savepath=os.path.join(save_dir,'psi_pdf.png'))
    plot_phi_vs_q_epsilon_sq(Q_prime, Q_hat_prime_pred, psi_pred, q, epsilon, savepath=os.path.join(save_dir,'phi_vs_q_epsilon_sq.png'))
    # Also run core advanced scientific plots:
    plot_trace_conservation(Q_hat_prime, Q_hat_prime_pred, savepath=os.path.join(save_dir,'trace_conservation.png'))
    plot_eigenvalue_distribution(Q_hat_prime, Q_hat_prime_pred, savepath=os.path.join(save_dir,'eigenvalue_dist.png'))
    plot_spectral_psd(Q_hat_prime_pred - Q_hat_prime, savepath=os.path.join(save_dir,'error_psd.png'))
    plot_energy_scale_consistency(epsilon, Q_prime, savepath=os.path.join(save_dir,'energy_scaling.png'))
    print("All advanced plots saved.")

# --- End of file ---
