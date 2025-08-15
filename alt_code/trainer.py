import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adamax, Adagrad, RAdam
try:
    from torch.optim import Adafactor
except ImportError:
    Adafactor = None
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os

from metrics.core_metrics import get_tensor_derivatives, process_ground_truth_Q, euler_angle_loss_detailed, magnitude_loss
from validate import evaluate_full  # Use your full validate.py!
# (Optionally import advanced_metrics for per-batch tracking or in-depth custom logic.)

def get_scheduler(optimizer, scheduler_type, epochs, warmup_steps=0, lr_min=1e-6, **kwargs):
    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    if scheduler_type == 'warmup_cosine':
        from transformers import get_cosine_schedule_with_warmup
        total_steps = epochs * kwargs.get('steps_per_epoch', 100)
        return get_cosine_schedule_with_warmup(optimizer, 
            num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=lr_min / optimizer.param_groups[0]['lr'], total_iters=epochs)
    if scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=kwargs.get('step_size', 30), gamma=kwargs.get('gamma', 0.1))
    return None

class Trainer:
    def __init__(
        self, 
        model_Q, model_psi, 
        train_set, val_set,
        batch_size=128, epochs=200,
        optimizer_Q='AdamW', optimizer_psi='AdamW',
        lr_Q=1e-3, lr_psi=1e-3,
        l1_lambda=0.0, l2_lambda=0.0,
        grad_clip=None,
        mixed_precision=True,
        device='cuda',
        scheduler_type='cosine',
        scheduler_warmup=0,
        scheduler_lr_min=1e-6,
        # Advanced options:
        beta1=0.9, beta2=0.999, beta_scheduler=None,
        teacher_Q=None, teacher_psi=None,   # for distillation/BYOL
        ema_decay=None,                     # if not None, use EMA weights
        swa_start=None,                     # epoch to start SWA
        swa_lr=None,                        # SWA learning rate
        checkpoint_dir='checkpoints',
        log_interval=10,
        save_every=10,
        history_savepath='history.npz',
        model_config=None,                  # dict for experimenting architectures
        run_advanced_every=5,
        architecture_tweaks=None,           # dict of model tweaks
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_Q = model_Q.to(self.device)
        self.model_psi = model_psi.to(self.device)
        self.epochs = epochs

        # --- Dataloaders ---
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        self.steps_per_epoch = len(self.train_loader)

        # --- Optimizers with L1/L2 regularization ---
        def param_groups(model):
            decay, no_decay = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad: continue
                if p.ndimension() >= 2 and 'bias' not in name:
                    decay.append(p)
                else:
                    no_decay.append(p)
            groups = [
                {'params': decay, 'weight_decay': l2_lambda},
                {'params': no_decay, 'weight_decay': 0.0}
            ]
            return groups

        def get_opt(opt_type, param_groups, lr, betas=(0.9,0.999)):
            opt_type = opt_type.lower()
            if opt_type == 'adamw':
                return AdamW(param_groups, lr=lr, betas=betas)
            if opt_type == 'adamax':
                return Adamax(param_groups, lr=lr, betas=betas)
            if opt_type == 'adagrad':
                return Adagrad(param_groups, lr=lr)
            if opt_type == 'radam':
                return RAdam(param_groups, lr=lr, betas=betas)
            if opt_type == 'adafactor' and Adafactor is not None:
                return Adafactor(param_groups, lr=lr)
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        self.optimizer_Q = get_opt(optimizer_Q, param_groups(self.model_Q), lr_Q, betas=(beta1, beta2))
        self.optimizer_psi = get_opt(optimizer_psi, param_groups(self.model_psi), lr_psi, betas=(beta1, beta2))

        # --- LR Schedulers ---
        self.scheduler_Q = get_scheduler(self.optimizer_Q, scheduler_type, epochs, warmup_steps=scheduler_warmup, lr_min=scheduler_lr_min, steps_per_epoch=self.steps_per_epoch)
        self.scheduler_psi = get_scheduler(self.optimizer_psi, scheduler_type, epochs, warmup_steps=scheduler_warmup, lr_min=scheduler_lr_min, steps_per_epoch=self.steps_per_epoch)

        # --- Mixed precision ---
        self.use_amp = mixed_precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # --- EMA/SWA ---
        self.ema_Q = AveragedModel(self.model_Q).to(self.device) if ema_decay else None
        self.ema_psi = AveragedModel(self.model_psi).to(self.device) if ema_decay else None
        self.ema_decay = ema_decay
        self.swa_Q = AveragedModel(self.model_Q).to(self.device) if swa_start else None
        self.swa_psi = AveragedModel(self.model_psi).to(self.device) if swa_start else None
        self.swa_start = swa_start
        self.swa_scheduler_Q = SWALR(self.optimizer_Q, swa_lr=swa_lr) if swa_start else None
        self.swa_scheduler_psi = SWALR(self.optimizer_psi, swa_lr=swa_lr) if swa_start else None

        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.log_interval = log_interval
        self.save_every = save_every
        self.history_savepath = history_savepath

        # --- Architecture tweaks, mods, configs ---
        self.model_config = model_config or {}
        self.architecture_tweaks = architecture_tweaks or {}

        # --- Advanced/epochic settings ---
        self.run_advanced_every = run_advanced_every

        # --- Loss history ---
        self.history = {
            'train_loss_Q': [], 'val_loss_Q': [],
            'train_loss_psi': [], 'val_loss_psi': [],
            # Add other core metrics as needed
        }

        # --- Distillation/teacher ---
        self.teacher_Q = teacher_Q
        self.teacher_psi = teacher_psi

    def save_checkpoint(self, epoch):
        state = {
            'model_Q': self.model_Q.state_dict(),
            'model_psi': self.model_psi.state_dict(),
            'optimizer_Q': self.optimizer_Q.state_dict(),
            'optimizer_psi': self.optimizer_psi.state_dict(),
            'scaler': self.scaler.state_dict() if self.use_amp else None,
            'epoch': epoch,
            'history': self.history,
        }
        if self.ema_Q: state['ema_Q'] = self.ema_Q.state_dict()
        if self.ema_psi: state['ema_psi'] = self.ema_psi.state_dict()
        if self.swa_Q: state['swa_Q'] = self.swa_Q.state_dict()
        if self.swa_psi: state['swa_psi'] = self.swa_psi.state_dict()
        path = os.path.join(self.checkpoint_dir, f'ckpt_epoch_{epoch}.pt')
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model_Q.load_state_dict(state['model_Q'])
        self.model_psi.load_state_dict(state['model_psi'])
        self.optimizer_Q.load_state_dict(state['optimizer_Q'])
        self.optimizer_psi.load_state_dict(state['optimizer_psi'])
        if self.use_amp and 'scaler' in state: self.scaler.load_state_dict(state['scaler'])
        self.history = state.get('history', self.history)
        if self.ema_Q and 'ema_Q' in state: self.ema_Q.load_state_dict(state['ema_Q'])
        if self.ema_psi and 'ema_psi' in state: self.ema_psi.load_state_dict(state['ema_psi'])
        if self.swa_Q and 'swa_Q' in state: self.swa_Q.load_state_dict(state['swa_Q'])
        if self.swa_psi and 'swa_psi' in state: self.swa_psi.load_state_dict(state['swa_psi'])
        print(f"Loaded checkpoint from {path}")

    def train(self):
        print(f"Training on {self.device} for {self.epochs} epochs.")
        for epoch in range(self.epochs):
            # === Training ===
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss_Q'].append(train_metrics['loss_Q'])
            self.history['train_loss_psi'].append(train_metrics['loss_psi'])

            # === Scheduler update (if batchwise, move inside train_epoch) ===
            if self.scheduler_Q: self.scheduler_Q.step()
            if self.scheduler_psi: self.scheduler_psi.step()

            # === EMA/SWA update ===
            if self.ema_Q:
                self.ema_Q.update_parameters(self.model_Q)
                self.ema_psi.update_parameters(self.model_psi)
            if self.swa_Q and epoch >= (self.swa_start or self.epochs+1):
                self.swa_Q.update_parameters(self.model_Q)
                self.swa_psi.update_parameters(self.model_psi)
                if self.swa_scheduler_Q: self.swa_scheduler_Q.step()
                if self.swa_scheduler_psi: self.swa_scheduler_psi.step()

            # === Save snapshot ===
            if (epoch+1) % self.save_every == 0:
                self.save_checkpoint(epoch+1)

            # === Per-epoch validation using validate.py! ===
            val_results = evaluate_full(
                model_Q=self.model_Q, model_psi=self.model_psi, dataset=self.val_loader.dataset,
                batch_size=self.val_loader.batch_size, device=str(self.device),
                run_advanced=(epoch+1) % self.run_advanced_every == 0,  # e.g. every N epochs
                plot_core=True,
                plot_advanced=(epoch+1) % self.run_advanced_every == 0,
                save_dir=os.path.join(self.checkpoint_dir, f'val_epoch_{epoch+1}'),
                max_batches=None,
                save_predictions=False,
                verbose=False,
            )
            self.history['val_loss_Q'].append(np.mean(val_results['loss_Q']))
            self.history['val_loss_psi'].append(np.mean(val_results['loss_psi']))

            print(f"Epoch {epoch+1:03d}: Train Q {train_metrics['loss_Q']:.4e}, Psi {train_metrics['loss_psi']:.4e} | "
                  f"Val Q {np.mean(val_results['loss_Q']):.4e}, Psi {np.mean(val_results['loss_psi']):.4e}")

            # (Optional: save all intermediate/final history)
            np.savez(self.history_savepath, **self.history)

        # === Final full validation ===
        print("\n=== FINAL FULL VALIDATION ===")
        final_val_dir = os.path.join(self.checkpoint_dir, "val_final")
        final_results = evaluate_full(
            model_Q=self.model_Q, model_psi=self.model_psi, dataset=self.val_loader.dataset,
            batch_size=self.val_loader.batch_size, device=str(self.device),
            run_advanced=True, plot_core=True, plot_advanced=True,
            save_dir=final_val_dir, max_batches=None, save_predictions=True, verbose=True,
        )
        print("Training and validation complete. Final figures and arrays saved.")

    def train_epoch(self, epoch):
        self.model_Q.train()
        self.model_psi.train()
        sum_loss_Q, sum_loss_psi = 0, 0
        for i, (A_batch, Q_batch) in enumerate(tqdm(self.train_loader, desc=f"Train {epoch+1}")):
            A_batch, Q_batch = A_batch.to(self.device), Q_batch.to(self.device)
            s, w, epsilon, q, r = get_tensor_derivatives(A_batch)
            _, Q_hat_prime_true, psi_true = process_ground_truth_Q(Q_batch, epsilon)
            a = A_batch / (epsilon.unsqueeze(-1).unsqueeze(-1) + 1e-8)
            s_norm = 0.5 * (a + a.transpose(1, 2))

            with autocast(enabled=self.use_amp):
                Q_hat_prime_pred = self.model_Q(s_norm, w)
                loss_Q, _ = euler_angle_loss_detailed(Q_hat_prime_pred, Q_hat_prime_true, s_norm)
                psi_pred = self.model_psi(q, r)
                loss_psi = magnitude_loss(psi_pred, psi_true)
                # Add L1/L2 regularization
                l1_penalty = sum(p.abs().sum() for p in self.model_Q.parameters() if p.requires_grad)
                l1_penalty += sum(p.abs().sum() for p in self.model_psi.parameters() if p.requires_grad)
                if self.architecture_tweaks.get('l1_on_Q_only', False):
                    l1_penalty = sum(p.abs().sum() for p in self.model_Q.parameters() if p.requires_grad)
                loss_Q_total = loss_Q + self.model_config.get('l1_lambda', 0.0) * l1_penalty

            self.optimizer_Q.zero_grad()
            self.optimizer_psi.zero_grad()
            self.scaler.scale(loss_Q_total).backward(retain_graph=True)
            self.scaler.scale(loss_psi).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer_Q)
                self.scaler.unscale_(self.optimizer_psi)
                torch.nn.utils.clip_grad_norm_(self.model_Q.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.model_psi.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer_Q)
            self.scaler.step(self.optimizer_psi)
            self.scaler.update()
            sum_loss_Q += loss_Q.item()
            sum_loss_psi += loss_psi.item()

            # Batchwise logging and diagnostics (optional/fine-tuning)

            if i % self.log_interval == 0 and i > 0:
                # ---- CORE METRICS (Batchwise) ----
                print(f"  Batch {i}/{len(self.train_loader)}: Loss Q={loss_Q.item():.3e}, Psi={loss_psi.item():.3e}")
                
                # Example: Similarity stats
                sim_Q = None
                sim_psi = None
                try:
                    from metrics.core_metrics import basic_similarity_metrics
                    sim_Q = basic_similarity_metrics(Q_hat_prime_pred, Q_hat_prime_true)
                    sim_psi = basic_similarity_metrics(psi_pred.unsqueeze(-1), psi_true.unsqueeze(-1))
                    print(f"    Q Pearson: {sim_Q['pearson_mean']:.3f}, NCC: {sim_Q['ncc_mean']:.3f}")
                    print(f"    Psi Pearson: {sim_psi['pearson_mean']:.3f}, NCC: {sim_psi['ncc_mean']:.3f}")
                except Exception as e:
                    pass  # For early debugging or if function not found

                # ---- ADVANCED METRICS (Per batch - expensive! Enable only if needed) ----
                if self.architecture_tweaks.get('advanced_metrics_batch', False):
                    try:
                        from metrics.advanced_metrics import AdvancedMetrics
                        # MC Dropout for Q
                        mc = AdvancedMetrics.mc_dropout(self.model_Q, (s_norm, w), mc_samples=5)
                        print(f"    Q MC Epistemic Uncertainty: {mc['epistemic_uncertainty']:.3e}")
                        # Optionally: adversarial, hessian, etc
                        # adv = AdvancedMetrics.adversarial(self.model_Q, (s_norm, w), Q_hat_prime_true)
                        # print(f"    Adversarial loss (Q): {adv['loss']:.3e}")
                    except Exception as e:
                        print("    (Advanced batch metrics error):", str(e))

                # ---- MINI-PLOTS (Debug/monitor) ----
                if self.architecture_tweaks.get('plot_mini_batch', False):
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(5,3))
                        plt.hist(Q_hat_prime_pred.detach().cpu().numpy().flatten(), bins=30, alpha=0.7, label='Pred')
                        plt.hist(Q_hat_prime_true.detach().cpu().numpy().flatten(), bins=30, alpha=0.7, label='True')
                        plt.title(f'Q_hat_prime values (Batch {i})')
                        plt.legend()
                        plt.show()
                        plt.close()
                    except Exception as e:
                        print("    (Mini-plot error):", str(e))


            if i % self.log_interval == 0 and i > 0:
                print(f"  Batch {i}/{len(self.train_loader)}: Loss Q={loss_Q.item():.3e}, Psi={loss_psi.item():.3e}")

        n_batches = len(self.train_loader)
        return {
            'loss_Q': sum_loss_Q / n_batches,
            'loss_psi': sum_loss_psi / n_batches,
        }

# --- End of trainer.py ---