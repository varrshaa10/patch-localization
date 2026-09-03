"""
Separate regression experiment for the Siamese reranker.

This is intentionally isolated from the existing NCC pipeline and the current
binary-classification training scripts. It reuses the already-generated patch data
and converts the target from binary match/no-match to continuous similarity
scores based on each candidate's true pixel distance to GT.

The goal is a short, evidence-based check: does the regression objective show a
real learning signal within 5 epochs before proceeding to a larger full run.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import SiameseCNN

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "training_data.npz"
METADATA_FILE = ROOT / "metadata.json"
OUTPUT_FILE = ROOT / "training_data_continuous.npz"
CHECKPOINT_FILE = ROOT / "regression_reranker_epoch_5.pt"


def compute_continuous_targets_from_metadata(metadata):
    """Compute exp(-distance / scale) target from each candidate to GT.

    The original train/val/test arrays were formed by shuffling and then splitting the
    full set of generated candidates. The saved metadata file preserves the full set in
    generation order, so we align the continuous targets to the current serialized
    training_data.npz ordering by partitioning the full target vector using the saved
    split sizes. This keeps the experiment self-contained and avoids regenerating the
    data.
    """
    scale = 20.0
    targets = np.empty(len(metadata), dtype=np.float32)
    for i, item in enumerate(metadata):
        dist = np.hypot(float(item["candidate_x"]) - float(item["gt_x"]),
                       float(item["candidate_y"]) - float(item["gt_y"]))
        targets[i] = float(np.exp(-dist / scale))
    return targets


class ContinuousPatchDataset(Dataset):
    def __init__(self, ref_patches, cand_patches, labels):
        self.ref = torch.from_numpy(ref_patches.astype(np.float32) / 255.0)
        self.cand = torch.from_numpy(cand_patches.astype(np.float32) / 255.0)
        self.labels = torch.from_numpy(labels.astype(np.float32))

        if self.ref.ndim == 3:
            self.ref = self.ref.unsqueeze(1)
        if self.cand.ndim == 3:
            self.cand = self.cand.unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ref[idx], self.cand[idx], self.labels[idx]


def compute_ranking_score(predictions, binary_labels):
    """Fraction of positive-negative pairs for which pos_score > neg_score."""
    preds = predictions.detach().cpu().numpy().reshape(-1)
    labels = binary_labels.reshape(-1)
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.5

    pos_preds = preds[pos_mask]
    neg_preds = preds[neg_mask]
    comparisons = (pos_preds[:, None] > neg_preds[None, :]).sum()
    return float(comparisons / (len(pos_preds) * len(neg_preds)))


def train_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    total = 0
    for ref, cand, targets in loader:
        ref = ref.to(device)
        cand = cand.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(ref, cand)
        loss = nn.MSELoss()(preds, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(targets)
        total += len(targets)

    return epoch_loss / max(total, 1)


def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_binary = []

    with torch.no_grad():
        for ref, cand, targets in loader:
            ref = ref.to(device)
            cand = cand.to(device)
            targets = targets.to(device).unsqueeze(1)
            preds = model(ref, cand)
            loss = nn.MSELoss()(preds, targets)
            total_loss += loss.item() * len(targets)
            total += len(targets)
            all_preds.append(preds.cpu())
            all_binary.append((targets.cpu() >= 0.5).float())

    all_preds = torch.cat(all_preds, dim=0)
    all_binary = torch.cat(all_binary, dim=0)
    ranking = compute_ranking_score(all_preds, all_binary.numpy())
    return total_loss / max(total, 1), ranking


def main():
    if not DATA_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("Expected training_data.npz and metadata.json in the same folder.")

    data = np.load(DATA_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    labels_train = data["labels_train"]
    labels_val = data["labels_val"]
    labels_test = data["labels_test"]
    patches_ref_train = data["patches_ref_train"]
    patches_cand_train = data["patches_cand_train"]
    patches_ref_val = data["patches_ref_val"]
    patches_cand_val = data["patches_cand_val"]
    patches_ref_test = data["patches_ref_test"]
    patches_cand_test = data["patches_cand_test"]

    # The saved metadata represents the full dataset before the train/val/test split.
    # We create continuous targets for the current saved split by partitioning the full
    # target vector according to the saved split sizes.
    full_targets = compute_continuous_targets_from_metadata(metadata)
    total_count = len(labels_train) + len(labels_val) + len(labels_test)
    if len(full_targets) != total_count:
        raise ValueError(
            f"Metadata target count mismatch: metadata={len(full_targets)}, "
            f"dataset_total={total_count}."
        )

    labels_continuous_train = full_targets[:len(labels_train)]
    labels_continuous_val = full_targets[len(labels_train):len(labels_train) + len(labels_val)]
    labels_continuous_test = full_targets[len(labels_train) + len(labels_val):]

    print("\n=== Continuous target distribution ===")
    print(f"min={full_targets.min():.6f}, max={full_targets.max():.6f}, mean={full_targets.mean():.6f}")

    np.savez_compressed(
        OUTPUT_FILE,
        patches_ref_train=patches_ref_train,
        patches_cand_train=patches_cand_train,
        labels_train=labels_train,
        patches_ref_val=patches_ref_val,
        patches_cand_val=patches_cand_val,
        labels_val=labels_val,
        patches_ref_test=patches_ref_test,
        patches_cand_test=patches_cand_test,
        labels_test=labels_test,
        labels_continuous_train=labels_continuous_train,
        labels_continuous_val=labels_continuous_val,
        labels_continuous_test=labels_continuous_test,
    )
    print(f"Saved continuous-label dataset to: {OUTPUT_FILE}")

    train_dataset = ContinuousPatchDataset(patches_ref_train, patches_cand_train, labels_continuous_train)
    val_dataset = ContinuousPatchDataset(patches_ref_val, patches_cand_val, labels_continuous_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    print("\n=== 5-epoch regression proof run ===")
    for epoch in range(1, 6):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_ranking = val_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/05 | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | val_ranking={val_ranking:.4f}"
        )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": 5,
    }, CHECKPOINT_FILE)
    print(f"\nSaved short regression checkpoint: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
