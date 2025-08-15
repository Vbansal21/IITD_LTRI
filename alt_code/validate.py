import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

from metrics.core_metrics import (
    get_tensor_derivatives, process_ground_truth_Q, euler_angle_loss_detailed, magnitude_loss,
    basic_similarity_metrics, angle_means
)
from metrics.advanced_metrics import AdvancedMetrics
from visualization.core_plots import (
    plot_losses, plot_cosine_angles, plot_prediction_similarity, plot_pdf, plot_joint_pdf
)
from visualization.advanced_plots import all_advanced_plots

def evaluate_full(
    model_Q, model_psi,
    dataset,
    batch_size=128,
    device='cuda',
    run_advanced=True,
    plot_core=True,
    plot_advanced=True,
    save_dir='figs',
    max_batches=None,
    save_predictions=False,
    verbose=True,
):
    """
    Full model evaluation, metric tracking, and figure generation.
    Saves all results for reproducibility and diagnostics.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_Q = model_Q.to(device)
    model_psi = model_psi.to(device)
    model_Q.eval(); model_psi.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = {
        'loss_Q': [], 'loss_psi': [],
        'similarity_Q': [], 'similarity_psi': [],
        'angle_stats': [],
        'mc_dropout_Q': [], 'mc_dropout_psi': [],
        'hessian_Q': [], 'adversarial_Q': [], 'adversarial_psi': [],
        'calibration_Q': [], 'calibration_psi': [],
        # For saving all predictions:
        'Q_hat_prime_true': [], 'Q_hat_prime_pred': [],
        'psi_true': [], 'psi_pred': [],
        's': [], 'w': [], 'epsilon': [], 'q': [], 'r': [],
        'Q_prime': [],
    }

    # --- Batchwise metrics over full dataset ---
    for i, (A_batch, Q_batch) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        A_batch, Q_batch = A_batch.to(device), Q_batch.to(device)
        s, w, epsilon, q, r = get_tensor_derivatives(A_batch)
        Q_prime, Q_hat_prime_true, psi_true = process_ground_truth_Q(Q_batch, epsilon)
        a = A_batch / (epsilon.unsqueeze(-1).unsqueeze(-1) + 1e-8)
        s_norm = 0.5 * (a + a.transpose(1, 2))
        Q_hat_prime_pred = model_Q(s_norm, w)
        psi_pred = model_psi(q, r)

        # --- Core metrics ---
        loss_Q, angle_comp = euler_angle_loss_detailed(Q_hat_prime_pred, Q_hat_prime_true, s_norm)
        loss_psi = magnitude_loss(psi_pred, psi_true)
        sim_Q = basic_similarity_metrics(Q_hat_prime_pred, Q_hat_prime_true)
        sim_psi = basic_similarity_metrics(psi_pred.unsqueeze(-1), psi_true.unsqueeze(-1))
        angles = angle_means(Q_hat_prime_pred, Q_hat_prime_true, s_norm)
        results['loss_Q'].append(loss_Q.item())
        results['loss_psi'].append(loss_psi.item())
        results['similarity_Q'].append(sim_Q)
        results['similarity_psi'].append(sim_psi)
        results['angle_stats'].append(angles)

        # --- Save for full-dataset advanced plots ---
        results['Q_hat_prime_true'].append(Q_hat_prime_true.cpu())
        results['Q_hat_prime_pred'].append(Q_hat_prime_pred.cpu())
        results['psi_true'].append(psi_true.cpu())
        results['psi_pred'].append(psi_pred.cpu())
        results['s'].append(s.cpu())
        results['w'].append(w.cpu())
        results['epsilon'].append(epsilon.cpu())
        results['q'].append(q.cpu())
        results['r'].append(r.cpu())
        results['Q_prime'].append(Q_prime.cpu())

        # --- Advanced metrics (last batch or all batches) ---
        if run_advanced:
            # MC Dropout
            mc_Q = AdvancedMetrics.mc_dropout(model_Q, (s_norm, w), mc_samples=20)
            mc_psi = AdvancedMetrics.mc_dropout(model_psi, (q, r), mc_samples=20)
            results['mc_dropout_Q'].append(mc_Q)
            results['mc_dropout_psi'].append(mc_psi)
            # Hessian (Q only, expensive!)
            hess_Q = AdvancedMetrics.hessian(model_Q, loss_Q)
            results['hessian_Q'].append(hess_Q)
            # Adversarial
            adv_Q = AdvancedMetrics.adversarial(model_Q, (s_norm, w), Q_hat_prime_true)
            adv_psi = AdvancedMetrics.adversarial(model_psi, (q, r), psi_true)
            results['adversarial_Q'].append(adv_Q)
            results['adversarial_psi'].append(adv_psi)
            # Calibration
            cal_bins_Q = AdvancedMetrics.calibration(
                mc_Q['mc_stds'].flatten(),
                (Q_hat_prime_pred - Q_hat_prime_true).norm(dim=(1,2)).cpu().numpy()
            )
            cal_bins_psi = AdvancedMetrics.calibration(
                mc_psi['mc_stds'].flatten(),
                (psi_pred - psi_true).abs().cpu().numpy()
            )
            results['calibration_Q'].append(cal_bins_Q)
            results['calibration_psi'].append(cal_bins_psi)

    # --- Aggregate arrays for full-dataset advanced plots ---
    arr = lambda k: torch.cat(results[k], dim=0).numpy()
    all_Q_hat_prime_true = arr('Q_hat_prime_true')
    all_Q_hat_prime_pred = arr('Q_hat_prime_pred')
    all_psi_true = arr('psi_true')
    all_psi_pred = arr('psi_pred')
    all_s = arr('s')
    all_w = arr('w')
    all_epsilon = arr('epsilon')
    all_q = arr('q')
    all_r = arr('r')
    all_Q_prime = arr('Q_prime')

    # --- CORE PLOTS ---
    if plot_core:
        plot_losses(results['loss_Q'], results['loss_psi'], savepath=f"{save_dir}/losses.png")
        plot_cosine_angles(results['angle_stats'], savepath=f"{save_dir}/cosine_angles.png")
        plot_prediction_similarity(results['similarity_Q'], results['similarity_psi'],
                                   savepath=f"{save_dir}/similarity.png")
        plot_pdf(all_psi_pred, all_psi_true, bins=60,
                 label_pred='Pred ψ', label_true='True ψ', title='PDF of ψ',
                 savepath=f"{save_dir}/psi_pdf.png")
        plot_joint_pdf(all_q, all_r, savepath=f"{save_dir}/qr_joint_pdf.png")

    # --- ADVANCED PLOTS ---
    if plot_advanced:
        all_advanced_plots(
            q=all_q, r=all_r,
            psi_true=all_psi_true, psi_pred=all_psi_pred,
            s=all_s, w=all_w,
            Q_prime=all_Q_prime,
            Q_hat_prime=all_Q_hat_prime_true,
            Q_hat_prime_pred=all_Q_hat_prime_pred,
            model_Q=model_Q,
            get_tensor_derivatives=get_tensor_derivatives,
            device=device,
            epsilon=all_epsilon,
            save_dir=save_dir
        )

    # --- Save all predictions/arrays if needed ---
    if save_predictions:
        np.savez(os.path.join(save_dir, "predictions.npz"),
                 Q_hat_prime_true=all_Q_hat_prime_true,
                 Q_hat_prime_pred=all_Q_hat_prime_pred,
                 psi_true=all_psi_true,
                 psi_pred=all_psi_pred,
                 s=all_s, w=all_w, epsilon=all_epsilon, q=all_q, r=all_r, Q_prime=all_Q_prime)

    # --- Print summary stats ---
    if verbose:
        print("==== VALIDATION SUMMARY ====")
        print(f"Q Loss (mean):   {np.mean(results['loss_Q']):.4e}")
        print(f"Psi Loss (mean): {np.mean(results['loss_psi']):.4e}")
        print(f"Q Pearson (mean, batch):  {np.mean([sim['pearson_mean'] for sim in results['similarity_Q']]):.3f}")
        print(f"Psi Pearson (mean, batch):{np.mean([sim['pearson_mean'] for sim in results['similarity_psi']]):.3f}")
        if len(results['mc_dropout_Q']) > 0:
            print(f"Q MC Dropout Epistemic Uncertainty: {results['mc_dropout_Q'][-1]['epistemic_uncertainty']:.3e}")
        if len(results['hessian_Q']) > 0:
            print(f"Q Hessian Trace Estimate: {results['hessian_Q'][-1]['hessian_trace_estimate']:.3e}")
        print("All diagnostic figures and results saved to:", save_dir)

    # --- Optional: save all results for later analysis ---
    import pickle
    with open(os.path.join(save_dir, 'validation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return results

# CLI usage:
if __name__ == "__main__":
    # from models import TBNN_Q_direction, FCNN_psi_magnitude
    # from datasets import MatlabTurbulenceDataset
    # model_Q = TBNN_Q_direction(); model_psi = FCNN_psi_magnitude()
    # dataset = MatlabTurbulenceDataset('velGrad.mat', 'PH.mat', device='cuda')
    # evaluate_full(model_Q, model_psi, dataset, device='cuda', save_dir='figs')
    pass
