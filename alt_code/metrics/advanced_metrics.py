# metrics/advanced_metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

def topk_hessian(model, loss, k=20):
    # Placeholder: for true top-k Hessian eigvals, use pyhessian, but for now, simple zeros
    try:
        from pyhessian import hessian
        h = hessian(model, loss)
        eigenvals, _ = h.eigenvalues(top_n=k)
        return [ev.item() for ev in eigenvals]
    except Exception:
        return [0.0] * k

class AdvancedMetrics:
    """
    Collection of advanced scientific diagnostics for turbulence NN.
    All methods are batch-agnostic; pass CPU or CUDA tensors as needed.
    """

    @staticmethod
    def hessian_metrics(model, loss):
        """Estimate Hessian trace and top eigenvals (approximate)."""
        try:
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.view(-1) for g in grads])
            # Hutchinson estimator for Hessian trace
            z = torch.randint_like(grad_vec, 0, 2).float() * 2 - 1
            hv = torch.autograd.grad(grad_vec, params, grad_outputs=z, retain_graph=True)
            hv_vec = torch.cat([h.view(-1) for h in hv])
            trace_estimate = torch.dot(z, hv_vec).item()
            grad_norm = torch.norm(grad_vec).item()
            hess_eigs = topk_hessian(model, loss, k=20)
            return {
                'hessian_trace_estimate': trace_estimate,
                'gradient_frobenius_norm': grad_norm,
                'hessian_top_eigvals': hess_eigs
            }
        except Exception:
            return {'hessian_trace_estimate': 0.0, 'gradient_frobenius_norm': 0.0, 'hessian_top_eigvals': [0.0]*20}

    @staticmethod
    def similarity(pred, true):
        """
        Pearson correlation & normalized cross-correlation (NCC) between prediction and true.
        Inputs: pred, true: (batch, *) torch.Tensor or np.ndarray
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(true, torch.Tensor):
            true = true.detach().cpu().numpy()
        pred_flat = pred.reshape(pred.shape[0], -1)
        true_flat = true.reshape(true.shape[0], -1)
        ncc, pearson = [], []
        for i in range(pred_flat.shape[0]):
            p, t = pred_flat[i], true_flat[i]
            if np.std(p) > 1e-8 and np.std(t) > 1e-8:
                try:
                    pearson.append(pearsonr(p, t)[0])
                except Exception: pearson.append(0)
                c = np.corrcoef(p, t)[0, 1]
                ncc.append(c if not np.isnan(c) else 0.0)
        return {
            'ncc_mean': np.mean(ncc) if ncc else 0.0,
            'ncc_std': np.std(ncc) if ncc else 0.0,
            'pearson_mean': np.mean(pearson) if pearson else 0.0,
            'pearson_std': np.std(pearson) if pearson else 0.0
        }

    @staticmethod
    def mc_dropout(model, inputs, mc_samples=20):
        """
        Run MC Dropout to estimate epistemic uncertainty.
        Inputs: model (in eval mode), inputs: tuple or tensor, mc_samples: int
        Returns: mean, std, predictive variance, epistemic uncertainty
        """
        # Make sure model uses dropout during inference!
        was_training = model.training
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                if isinstance(inputs, tuple):
                    preds.append(model(*inputs).detach().cpu().numpy())
                else:
                    preds.append(model(inputs).detach().cpu().numpy())
        model.train(was_training)
        preds = np.stack(preds, axis=0)  # (mc, batch, ...)
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        ep_unc = std.mean()
        pred_var = (std**2).mean()
        return {
            'mc_mean': mean,
            'mc_std': std,
            'epistemic_uncertainty': ep_unc,
            'predictive_variance': pred_var,
            'mc_stds': std
        }

    @staticmethod
    def adversarial(model, inputs, targets, epsilon=0.01):
        """
        Compute adversarial (FGSM) loss delta.
        Returns: adversarial inputs, adv loss
        """
        # Clone and set requires_grad
        def grad_input(x): return x.clone().detach().requires_grad_(True)
        if isinstance(inputs, tuple):
            inps = tuple(grad_input(x) for x in inputs)
        else:
            inps = grad_input(inputs)
        model.eval()
        if isinstance(inps, tuple):
            outputs = model(*inps)
        else:
            outputs = model(inps)
        loss = F.mse_loss(outputs.view(-1), targets.view(-1))
        model.zero_grad()
        loss.backward(retain_graph=True)
        if isinstance(inps, tuple):
            adv_inputs = tuple(x + epsilon * x.grad.sign() for x in inps)
            adv_out = model(*adv_inputs)
        else:
            adv_inputs = inps + epsilon * inps.grad.sign()
            adv_out = model(adv_inputs)
        adv_loss = F.mse_loss(adv_out.view(-1), targets.view(-1)).item()
        return {
            'adv_inputs': adv_inputs,
            'adv_loss': adv_loss
        }

    @staticmethod
    def input_saliency(model, inputs, outputs):
        """
        Saliency = gradient of output w.r.t. input, mean abs val.
        """
        if isinstance(inputs, tuple):
            for x in inputs:
                x.requires_grad_(True)
            output_sum = outputs.sum()
            grads = torch.autograd.grad(output_sum, inputs, retain_graph=True)
            return {
                'input_saliency': [g.abs().mean().item() for g in grads]
            }
        else:
            inputs.requires_grad_(True)
            output_sum = outputs.sum()
            grad = torch.autograd.grad(output_sum, inputs, retain_graph=True)[0]
            return {
                'input_saliency': grad.abs().mean().item()
            }

    @staticmethod
    def calibration(uncertainty, error, bins=10):
        """
        Returns binned (uncertainty, error) for reliability diagram.
        Inputs: uncertainty, error: 1D arrays
        """
        unc = np.asarray(uncertainty)
        err = np.asarray(error)
        order = np.argsort(unc)
        unc, err = unc[order], err[order]
        n = len(unc)
        bin_size = n // bins
        bin_unc, bin_err = [], []
        for i in range(bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < bins - 1 else n
            if end > start:
                bin_unc.append(unc[start:end].mean())
                bin_err.append(err[start:end].mean())
        return np.array(bin_unc), np.array(bin_err)

# --- End of file ---
