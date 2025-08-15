# visualization/core_plots.py

import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_loss_Q, val_loss_Q, train_loss_psi, val_loss_psi, savepath=None):
    """Plot training/validation losses for direction and magnitude models."""
    epochs = np.arange(1, len(train_loss_Q)+1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss_Q, label='Train Q', lw=2)
    plt.plot(epochs, val_loss_Q, label='Val Q', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Q direction)')
    plt.title('Q Model Loss')
    plt.yscale('log')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_loss_psi, label='Train ψ', lw=2)
    plt.plot(epochs, val_loss_psi, label='Val ψ', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (ψ magnitude)')
    plt.title('Psi Model Loss')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_cosine_angles(
    train_cos_zeta_true, train_cos_zeta_pred, train_cos_theta_true, train_cos_theta_pred,
    val_cos_zeta_true, val_cos_zeta_pred, val_cos_theta_true, val_cos_theta_pred,
    train_cos_eta_true, train_cos_eta_pred, val_cos_eta_true, val_cos_eta_pred,
    savepath=None
):
    """Plot cosine angle means for zeta/theta/eta, train and val (true vs pred)."""
    epochs = np.arange(1, len(train_cos_zeta_true)+1)
    plt.figure(figsize=(12, 8))

    # Zeta (largest Q'/s eigenvector alignment)
    plt.subplot(3,2,1)
    plt.plot(epochs, train_cos_zeta_true, label='Train True', lw=2)
    plt.plot(epochs, train_cos_zeta_pred, label='Train Pred', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('|cos ζ|')
    plt.title('Train: |cos(zeta)|')
    plt.legend()

    plt.subplot(3,2,2)
    plt.plot(epochs, val_cos_zeta_true, label='Val True', lw=2)
    plt.plot(epochs, val_cos_zeta_pred, label='Val Pred', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('|cos ζ|')
    plt.title('Val: |cos(zeta)|')
    plt.legend()

    # Theta (2nd eigenvector alignment)
    plt.subplot(3,2,3)
    plt.plot(epochs, train_cos_theta_true, label='Train True', lw=2)
    plt.plot(epochs, train_cos_theta_pred, label='Train Pred', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('|cos θ|')
    plt.title('Train: |cos(theta)|')
    plt.legend()

    plt.subplot(3,2,4)
    plt.plot(epochs, val_cos_theta_true, label='Val True', lw=2)
    plt.plot(epochs, val_cos_theta_pred, label='Val Pred', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('|cos θ|')
    plt.title('Val: |cos(theta)|')
    plt.legend()

    # Eta (3rd eigenvector alignment)
    plt.subplot(3,2,5)
    plt.plot(epochs, train_cos_eta_true, label='Train True', lw=2)
    plt.plot(epochs, train_cos_eta_pred, label='Train Pred', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('|cos η|')
    plt.title('Train: |cos(eta)|')
    plt.legend()

    plt.subplot(3,2,6)
    plt.plot(epochs, val_cos_eta_true, label='Val True', lw=2)
    plt.plot(epochs, val_cos_eta_pred, label='Val Pred', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('|cos η|')
    plt.title('Val: |cos(eta)|')
    plt.legend()

    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_prediction_similarity(
    train_pearson_Q, val_pearson_Q, train_ncc_Q, val_ncc_Q,
    train_pearson_psi, val_pearson_psi, train_ncc_psi, val_ncc_psi,
    savepath=None
):
    """Plot prediction similarity metrics (Pearson/NCC) for Q and psi models."""
    epochs = np.arange(1, len(train_pearson_Q)+1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_pearson_Q, label='Train Q', lw=2)
    plt.plot(epochs, val_pearson_Q, label='Val Q', lw=2)
    plt.plot(epochs, train_ncc_Q, label='Train Q (NCC)', lw=2, ls='--')
    plt.plot(epochs, val_ncc_Q, label='Val Q (NCC)', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('Similarity')
    plt.title('Prediction Similarity: Q Model')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_pearson_psi, label='Train ψ', lw=2)
    plt.plot(epochs, val_pearson_psi, label='Val ψ', lw=2)
    plt.plot(epochs, train_ncc_psi, label='Train ψ (NCC)', lw=2, ls='--')
    plt.plot(epochs, val_ncc_psi, label='Val ψ (NCC)', lw=2, ls='--')
    plt.xlabel('Epoch'); plt.ylabel('Similarity')
    plt.title('Prediction Similarity: Psi Model')
    plt.legend()
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()

def plot_pdf(pred, true, bins=50, label_pred='Predicted', label_true='True', title='PDF', savepath=None):
    """Plot PDFs for predictions and targets (e.g. psi, Q' norm)."""
    plt.figure(figsize=(6,4))
    plt.hist(true, bins=bins, alpha=0.7, label=label_true, density=True)
    plt.hist(pred, bins=bins, alpha=0.7, label=label_pred, density=True, ls='--', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    plt.show()
