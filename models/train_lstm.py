"""
A-Insight Pro
LSTM Model Training V2.0
- Walk-forward expanding window training
- Early stopping with patience + ReduceLROnPlateau
- Gradient clipping + weight decay
- Better validation metrics (directional accuracy)
- Saves feature columns for ensemble prediction
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODEL_DIR, get_lstm_cfg
from models.lstm_model import StockLSTM, HuberLoss
from models.dataset import StockSequenceDataset, load_all_features, get_available_features, split_by_date


CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).squeeze(-1)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    """Validation step with directional accuracy."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze(-1)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    corr = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0

    # Directional accuracy
    dir_acc = np.mean((all_preds > 0) == (all_targets > 0)) if len(all_preds) > 0 else 0

    return avg_loss, mae, corr, dir_acc


def train_lstm():
    """Main training function."""
    print("=" * 60)
    print("LSTM Model Training V2.0")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Config
    cfg = get_lstm_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n[1/5] Loading features...")
    df = load_all_features()
    feature_cols = get_available_features(df)
    n_features = len(feature_cols)
    print(f"  Total features available: {n_features}")

    # Split by date
    print("\n[2/5] Splitting data (time-ordered)...")
    train_df, val_df, test_df = split_by_date(df)

    # Create datasets
    seq_len = cfg.get("seq_len", 60)
    print(f"\n[3/5] Creating datasets (seq_len={seq_len})...")
    train_dataset = StockSequenceDataset(train_df, seq_len=seq_len, feature_cols=feature_cols)
    val_dataset = StockSequenceDataset(val_df, seq_len=seq_len, feature_cols=feature_cols)

    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print(f"\n[4/5] Building model...")
    model = StockLSTM(
        n_features=n_features,
        hidden_size_1=cfg.get("hidden_size_1", 128),
        hidden_size_2=cfg.get("hidden_size_2", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
        attention_heads=cfg.get("attention_heads", 4)
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 0.001),
        weight_decay=cfg.get("weight_decay", 0.0001)
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        min_lr=1e-6, verbose=True
    )

    # Loss
    huber_delta = cfg.get("huber_delta", 1.0)
    criterion = HuberLoss(delta=huber_delta)
    print(f"  Huber delta: {huber_delta}")
    print(f"  LR scheduler: ReduceLROnPlateau(factor=0.5, patience=5)")

    # Training loop
    print(f"\n[5/5] Training...")
    epochs = cfg.get("epochs", 100)
    patience = cfg.get("early_stopping_patience", 15)
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "lstm_best.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_corr, val_dir_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"MAE: {val_mae:.4f} | "
                  f"Corr: {val_corr:.4f} | "
                  f"DirAcc: {val_dir_acc:.3f} | "
                  f"LR: {current_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_corr": val_corr,
                "val_dir_acc": val_dir_acc,
                "feature_cols": feature_cols,
                "n_features": n_features,
                "config": cfg,
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best model
    print(f"\n  Loading best model (epoch with val_loss={best_val_loss:.6f})...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Best epoch: {checkpoint['epoch']}, val_mae={checkpoint['val_mae']:.4f}, "
          f"val_dir_acc={checkpoint['val_dir_acc']:.3f}")

    # Test evaluation
    print("\n  Running final test evaluation...")
    test_dataset = StockSequenceDataset(test_df, seq_len=seq_len, feature_cols=feature_cols)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_mae, test_corr, test_dir_acc = validate(model, test_loader, criterion, device)
    print(f"  Test — Loss: {test_loss:.6f}, MAE: {test_mae:.4f}, "
          f"Corr: {test_corr:.4f}, DirAcc: {test_dir_acc:.3f}")

    # Save model + metadata
    model_file = os.path.join(MODEL_DIR, "stock_model_lstm.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "n_features": n_features,
        "config": cfg,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_corr": test_corr,
        "test_dir_acc": test_dir_acc,
        "training_version": "V2.0",
    }, model_file)
    print(f"\n  Model saved: {model_file}")

    # Save feature list for prediction
    with open(os.path.join(MODEL_DIR, "feature_cols.txt"), "w") as f:
        f.write("\n".join(feature_cols))
    print(f"  Feature list saved: models/feature_cols.txt")

    print("=" * 60)
    print("LSTM Training V2.0 Complete")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test DirAcc: {test_dir_acc:.3f}")
    print("=" * 60)

    return model, test_mae


if __name__ == "__main__":
    train_lstm()
