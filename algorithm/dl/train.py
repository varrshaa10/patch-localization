"""
Training script for Siamese CNN reranker model.

Features:
- Weighted cross-entropy loss (7.5x weight for positive class)
- Batch-level oversampling of positives (target 30/70 balance per batch)
- Early stopping on validation loss plateau
- Checkpointing best model

Usage:
    python train.py --epochs 5
    python train.py --epochs 100 --batch_size 32 --learning_rate 0.001
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import SiameseCNN, WeightedBCELoss


class PatchMatchDataset(Dataset):
    """Dataset of (reference_patch, candidate_patch, label) triplets."""
    
    def __init__(self, ref_patches, cand_patches, labels):
        """
        Args:
            ref_patches: (N, 128, 128) numpy array, uint8
            cand_patches: (N, 128, 128) numpy array, uint8
            labels: (N,) numpy array, {0, 1}
        """
        self.ref_patches = torch.from_numpy(ref_patches).float() / 255.0
        self.cand_patches = torch.from_numpy(cand_patches).float() / 255.0
        self.labels = torch.from_numpy(labels).float()
        
        # Add channel dimension if needed
        if self.ref_patches.ndim == 3:
            self.ref_patches = self.ref_patches.unsqueeze(1)
        if self.cand_patches.ndim == 3:
            self.cand_patches = self.cand_patches.unsqueeze(1)
        
        assert self.ref_patches.shape[0] == self.cand_patches.shape[0] == self.labels.shape[0]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.ref_patches[idx],  # (1, 128, 128)
            self.cand_patches[idx],  # (1, 128, 128)
            self.labels[idx]  # scalar
        )


class BalancedBatchSampler:
    """
    Custom batch sampler that oversamples positive examples.
    Target: ~30% positive, 70% negative per batch.
    """
    
    def __init__(self, labels, batch_size, target_pos_ratio=0.30):
        self.labels = labels
        self.batch_size = batch_size
        self.target_pos_ratio = target_pos_ratio
        
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]
        
        print(f"BalancedBatchSampler: {len(self.pos_indices)} positives, "
              f"{len(self.neg_indices)} negatives")
        print(f"Target per batch: {int(batch_size * target_pos_ratio)} pos, "
              f"{int(batch_size * (1 - target_pos_ratio))} neg")
    
    def __iter__(self):
        # Shuffle indices
        pos_shuffled = np.random.permutation(self.pos_indices)
        neg_shuffled = np.random.permutation(self.neg_indices)
        
        num_batches = len(self.labels) // self.batch_size
        
        for batch_idx in range(num_batches):
            # Calculate how many positives and negatives for this batch
            num_pos = int(self.batch_size * self.target_pos_ratio)
            num_neg = self.batch_size - num_pos
            
            # Sample with replacement from positives (if not enough)
            if batch_idx * num_pos < len(pos_shuffled):
                batch_pos = pos_shuffled[batch_idx * num_pos:(batch_idx + 1) * num_pos]
            else:
                batch_pos = np.random.choice(self.pos_indices, size=num_pos, replace=True)
            
            # Sample with replacement from negatives
            if batch_idx * num_neg < len(neg_shuffled):
                batch_neg = neg_shuffled[batch_idx * num_neg:(batch_idx + 1) * num_neg]
            else:
                batch_neg = np.random.choice(self.neg_indices, size=num_neg, replace=True)
            
            # Combine and shuffle
            batch = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch)
            
            yield batch.tolist()
    
    def __len__(self):
        return len(self.labels) // self.batch_size


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for ref_patches, cand_patches, labels in train_loader:
        ref_patches = ref_patches.to(device)
        cand_patches = cand_patches.to(device)
        labels = labels.to(device).unsqueeze(1)  # (batch_size, 1)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(ref_patches, cand_patches)  # (batch_size, 1)
        loss = loss_fn(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * labels.size(0)
        predictions_binary = (predictions >= 0.5).float()
        total_correct += (predictions_binary == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def val_epoch(model, val_loader, loss_fn, device):
    """Validate for one epoch. Returns loss, accuracy, and ranking score."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for ref_patches, cand_patches, labels in val_loader:
            ref_patches = ref_patches.to(device)
            cand_patches = cand_patches.to(device)
            labels = labels.to(device).unsqueeze(1)  # (batch_size, 1)
            
            predictions = model(ref_patches, cand_patches)
            loss = loss_fn(predictions, labels)
            
            total_loss += loss.item() * labels.size(0)
            predictions_binary = (predictions >= 0.5).float()
            total_correct += (predictions_binary == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.append(predictions.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    # Compute ranking score: fraction of (pos, neg) pairs where pred_pos > pred_neg
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        preds_pos = all_preds[pos_mask]
        preds_neg = all_preds[neg_mask]
        
        # Fraction of pos > neg pairs
        ranking_score = 0.0
        for p_pred in preds_pos:
            ranking_score += (p_pred > preds_neg).sum()
        ranking_score /= (len(preds_pos) * len(preds_neg))
    else:
        ranking_score = 0.5
    
    return avg_loss, avg_acc, ranking_score


def train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, checkpoint_dir):
    """Train model with early stopping based on ranking score (not loss)."""
    
    best_ranking_score = -1.0
    patience = 8
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_ranking_score": [],
        "epoch_times": []
    }
    
    print(f"\nTraining for up to {num_epochs} epochs with early stopping (patience={patience})...")
    print(f"Early stopping based on RANKING SCORE (highest ranking = best model).")
    print(f"Checkpoints saved every 5 epochs and at best ranking score.")
    print("=" * 100)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_ranking_score = val_epoch(model, val_loader, loss_fn, device)
        
        epoch_time = time.time() - start_time
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ranking_score"].append(val_ranking_score)
        history["epoch_times"].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
              f"TrLoss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"VLoss: {val_loss:.4f} Acc: {val_acc:.3f} RankScore: {val_ranking_score:.3f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Check for improvement in RANKING SCORE (not loss)
        if val_ranking_score > best_ranking_score:
            best_ranking_score = val_ranking_score
            patience_counter = 0
            
            # Save best checkpoint
            best_checkpoint_path = checkpoint_dir / "reranker_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ranking_score": val_ranking_score,
                "history": history
            }, best_checkpoint_path)
            print(f"  ✓ Best model saved (ranking: {best_ranking_score:.3f}, val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            periodic_checkpoint_path = checkpoint_dir / f"reranker_model_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ranking_score": val_ranking_score,
                "history": history
            }, periodic_checkpoint_path)
            print(f"  ✓ Periodic checkpoint saved: epoch_{epoch + 1}.pt")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (no improvement for {patience} epochs)")
            break
    
    print("=" * 100)
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Siamese CNN reranker")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device: 'cuda', 'cpu', or 'auto' (use GPU if available)")
    parser.add_argument("--checkpoint_dir", type=str, default=".",
                       help="Directory to save checkpoints")
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print(f"\nLoading training data from training_data.npz...")
    data = np.load("training_data.npz")
    
    patches_ref_train = data["patches_ref_train"]
    patches_cand_train = data["patches_cand_train"]
    labels_train = data["labels_train"]
    
    patches_ref_val = data["patches_ref_val"]
    patches_cand_val = data["patches_cand_val"]
    labels_val = data["labels_val"]
    
    print(f"  Train: {len(labels_train)} samples "
          f"({(labels_train == 1).sum()} pos, {(labels_train == 0).sum()} neg)")
    print(f"  Val:   {len(labels_val)} samples "
          f"({(labels_val == 1).sum()} pos, {(labels_val == 0).sum()} neg)")
    
    # Create datasets
    train_dataset = PatchMatchDataset(patches_ref_train, patches_cand_train, labels_train)
    val_dataset = PatchMatchDataset(patches_ref_val, patches_cand_val, labels_val)
    
    # Create data loaders with balanced batch sampling
    balanced_sampler = BalancedBatchSampler(
        labels_train,
        batch_size=args.batch_size,
        target_pos_ratio=0.30
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=balanced_sampler,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nBatches per epoch: {len(train_loader)}")
    
    # Model, optimizer, loss
    model = SiameseCNN().to(device)
    print(f"\nModel created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = WeightedBCELoss(pos_weight=7.5)
    
    # Train
    start_time = time.time()
    history = train(
        model, train_loader, val_loader, optimizer, loss_fn,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    total_time = time.time() - start_time
    
    # Save history
    history_file = checkpoint_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Epochs completed: {len(history['train_loss'])}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}, acc: {history['train_acc'][-1]:.3f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}, acc: {history['val_acc'][-1]:.3f}, "
          f"ranking: {history['val_ranking_score'][-1]:.3f}")
    print(f"\nBest ranking score: {max(history['val_ranking_score']):.3f}")
    best_ranking_idx = np.argmax(history['val_ranking_score'])
    print(f"  -> Best epoch: {best_ranking_idx + 1}, val_loss: {history['val_loss'][best_ranking_idx]:.4f}")
    window = min(10, len(history['val_ranking_score']))
    recent_scores = history['val_ranking_score'][-window:]
    recent_epochs = list(range(len(history['val_ranking_score']) - window + 1, len(history['val_ranking_score']) + 1))
    print(f"  -> Last {window} epochs (epochs {recent_epochs[0]}-{recent_epochs[-1]}): {recent_scores}")
    print(f"  -> Checkpoint saved as: {checkpoint_dir / 'reranker_model.pt'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
