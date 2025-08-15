# main.py
import os
import argparse
import torch
import numpy as np
import random

from models import TBNN_Q_direction, FCNN_psi_magnitude  # or import alternatives
from datasets import MatlabTurbulenceDataset, get_split_indices
from trainer import Trainer
from validate import evaluate_full

# === Argument parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="Turbulence AIBM Neural Model Training Framework")
    parser.add_argument('--velgrad_mat', type=str, required=True, help="Path to velGrad.mat file")
    parser.add_argument('--ph_mat', type=str, required=True, help="Path to PH.mat file")
    parser.add_argument('--device', type=str, default='cuda', help="Device for training (cuda/cpu)")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_Q', type=float, default=1e-3)
    parser.add_argument('--lr_psi', type=float, default=5e-3)
    parser.add_argument('--optimizer_Q', type=str, default='AdamW')
    parser.add_argument('--optimizer_psi', type=str, default='AdamW')
    parser.add_argument('--weight_decay_L2', type=float, default=1e-6)
    parser.add_argument('--weight_decay_L1', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'linear', 'none'])
    parser.add_argument('--ema_decay', type=float, default=None)
    parser.add_argument('--swa_start', type=int, default=None)
    parser.add_argument('--swa_lr', type=float, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--fig_dir', type=str, default='figs')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch_mod', type=str, default='', help="Optional: model arch modifiers")
    parser.add_argument('--train_val_split', type=float, default=0.85)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--advanced_metrics', action='store_true')
    parser.add_argument('--plot_core', action='store_true')
    parser.add_argument('--plot_advanced', action='store_true')
    parser.add_argument('--final_only_adv', action='store_true', help="Advanced metrics/plots only on final eval")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument('--inference', action='store_true', help="Run in inference mode only")
    parser.add_argument('--no_train', action='store_true', help="Only validate/evaluate, no training")
    # Add more arguments as needed
    return parser.parse_args()

# === Main execution logic ===
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"===> Loading Dataset ({args.velgrad_mat}, {args.ph_mat})")
    dataset = MatlabTurbulenceDataset(args.velgrad_mat, args.ph_mat, device=device)
    n_total = len(dataset)
    n_train = int(args.train_val_split * n_total)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:n_train], indices[n_train:]
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    print(f"Train/val split: {n_train}/{n_total - n_train}")

    # === Model instantiation (configurable) ===
    print("===> Building models...")
    # You may want to parse arch_mod string for options (deeper, wider, dropouts, etc)
    tbnn_hidden = [int(x) for x in os.environ.get('TBNN_HIDDEN', '50,100,100,100,50').split(',')]
    fcnn_hidden = [int(x) for x in os.environ.get('FCNN_HIDDEN', '50,80,50').split(',')]
    tbnn_dropout = float(os.environ.get('TBNN_DROPOUT', 0.1))
    fcnn_dropout = float(os.environ.get('FCNN_DROPOUT', 0.1))
    model_Q = TBNN_Q_direction(hidden_layers=tbnn_hidden, dropout_p=tbnn_dropout)
    model_psi = FCNN_psi_magnitude(hidden_layers=fcnn_hidden, dropout_p=fcnn_dropout)

    # === Optimizer configs ===
    opt_kwargs_Q = dict(weight_decay=args.weight_decay_L2)
    opt_kwargs_psi = dict(weight_decay=args.weight_decay_L2)
    # (optionally add L1 regularization via closure if needed)

    # === Trainer class ===
    trainer = Trainer(
        model_Q=model_Q, model_psi=model_psi,
        train_set=train_set, val_set=val_set,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer_Q=args.optimizer_Q,
        optimizer_psi=args.optimizer_psi,
        lr_Q=args.lr_Q,
        lr_psi=args.lr_psi,
        grad_clip=None,
        mixed_precision=True,
        device=args.device,
        ema_decay=args.ema_decay,
        swa_start=args.swa_start,
        swa_lr=args.swa_lr,
        checkpoint_dir=args.checkpoint_dir,
        advanced_metrics=args.advanced_metrics,
        run_adv_last_only=args.final_only_adv,
        log_interval=args.log_interval,
        save_every=args.save_every,
        # You can pass arch tweaks, teacher/student config, etc here as kwargs
        architecture_tweaks={
            'advanced_metrics_batch': False,
            'plot_mini_batch': False,
        }
    )

    # === Resume checkpoint (optional) ===
    if args.resume:
        print(f"===> Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # === TRAINING ===
    if not args.no_train and not args.inference:
        print("===> Starting training loop...")
        trainer.train()
        print("===> Training complete.")
        # Save final models
        trainer.save_checkpoint('final')
    else:
        print("===> Skipping training (validation/inference mode)")

    # === FINAL VALIDATION & FIGURES ===
    print("===> Running full validation and generating figures...")
    evaluate_full(
        model_Q=trainer.model_Q,
        model_psi=trainer.model_psi,
        dataset=val_set,
        batch_size=args.batch_size,
        device=args.device,
        run_advanced=args.advanced_metrics or args.final_only_adv,
        plot_core=args.plot_core,
        plot_advanced=args.plot_advanced,
        save_dir=args.fig_dir
    )

    print(f"===> All done! Results, checkpoints, and figures saved in '{args.fig_dir}' and '{args.checkpoint_dir}'.")

if __name__ == "__main__":
    main()