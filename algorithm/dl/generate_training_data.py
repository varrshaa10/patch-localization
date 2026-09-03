"""
Generate training data for NCC candidate re-ranker model.

Generates 500 pairs with 70% no-marker (periodic ambiguous) and 30% marker cases.
For each pair:
  - Runs existing ncc_match_multi() to get top 5-8 candidates
  - Crops local patches from search image at candidate locations
  - Labels each: 1 if within ~5 pixels of ground truth, 0 otherwise
  - Saves as .npz with 70/15/15 train/val/test split

Output: algorithm/dl/training_data.npz
  - patches_train, labels_train
  - patches_val, labels_val
  - patches_test, labels_test
  - metadata (pair_ids, candidate_locations, etc.)
"""

import sys
import os
import time
import json
import random
import argparse
import csv
from pathlib import Path

import numpy as np
import cv2

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from generate_dataset import generate_pair, REF_SIZE, SEARCH_SIZE, DOWNSAMPLE
from ncc import ncc_match_multi


def label_candidate(candidate_x, candidate_y, gt_x, gt_y, threshold_px=5.0):
    """
    Label a candidate as 1 (positive) if within threshold_px of ground truth,
    0 (negative) otherwise.
    """
    dist = np.sqrt((candidate_x - gt_x)**2 + (candidate_y - gt_y)**2)
    return 1 if dist <= threshold_px else 0


def crop_patch_at_location(image, cx, cy, patch_size=None):
    """
    Crop a patch centered at (cx, cy) from the image.
    If patch_size is None, use the reference size from dataset params.
    """
    if patch_size is None:
        patch_size = REF_SIZE
    
    h, w = image.shape[:2]
    x0 = max(0, int(cx - patch_size // 2))
    y0 = max(0, int(cy - patch_size // 2))
    x1 = min(w, int(cx + patch_size // 2))
    y1 = min(h, int(cy + patch_size // 2))
    
    # Crop the patch
    patch = image[y0:y1, x0:x1]
    
    # If patch is smaller than patch_size (near edge), pad it
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded = np.ones((patch_size, patch_size), dtype=image.dtype) * 30
        py0 = max(0, int(patch_size // 2 - (cy - 0)))
        px0 = max(0, int(patch_size // 2 - (cx - 0)))
        padded[py0:int(py0+patch.shape[0]), px0:int(px0+patch.shape[1])] = patch
        patch = padded
    
    return patch


def generate_training_data(num_pairs=500, output_dir="../tests/synthetic_data", 
                          train_split=0.70, val_split=0.15, ref_patch_size=128):
    """
    Generate training data with labeled candidates.
    
    Args:
      num_pairs: Number of synthetic pairs to generate
      output_dir: Directory where generated pairs are stored
      train_split: Fraction for training set
      val_split: Fraction for validation set
      ref_patch_size: Size of reference patch for NCC matching (NOT the full 1000x1000)
    
    Returns:
      patches_train, labels_train
      patches_val, labels_val
      patches_test, labels_test
      metadata dict
    """
    
    start_time = time.time()
    print("\n" + "="*70)
    print("STEP 1: Generate Training Data with Labeled Candidates")
    print("="*70 + "\n")
    
    print(f"Generating {num_pairs} pairs with 70% no-marker, 30% marker...")
    
    # Setup output directories
    training_pairs_dir = os.path.join(output_dir, "training_pairs")
    os.makedirs(training_pairs_dir, exist_ok=True)
    
    all_patches_ref = []  # Reference patches
    all_patches_cand = []  # Candidate patches
    all_labels = []
    all_metadata = []
    
    label_counts = {"positive": 0, "negative": 0}
    
    # Generate pairs
    for i in range(num_pairs):
        # Weight toward no-marker (70% no-marker, 30% marker)
        # First 70% of pairs: no marker; last 30%: marker
        add_marker = i >= int(num_pairs * 0.70)
        
        # Generate pair
        pair_meta = generate_pair(
            architecture=["dram", "finfet"][i % 2],
            pair_id=i,
            out_dir=training_pairs_dir,
            add_marker=add_marker,
            mode="L",
            jitter=False
        )
        
        # Load reference and search images
        ref_path = os.path.join(training_pairs_dir, f"pair_{i}", "reference.png")
        search_path = os.path.join(training_pairs_dir, f"pair_{i}", "search.png")
        
        reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        if reference is None or search is None:
            print(f"  Warning: Could not load images for pair {i}, skipping.")
            continue
        
        gt_x = pair_meta["gt_x"]
        gt_y = pair_meta["gt_y"]
        
        # Extract a smaller reference patch centered at ground truth location
        # for NCC matching (NOT the full 1000x1000 from original generation)
        reference_ncc = crop_patch_at_location(search, gt_x, gt_y, patch_size=ref_patch_size)
        
        # Run NCC matching to get top 5-8 candidates
        try:
            candidates = ncc_match_multi(search, reference_ncc, num_peaks=8, min_distance=15)
        except Exception as e:
            print(f"  Warning: NCC matching failed for pair {i}: {e}")
            continue
        
        if not candidates:
            print(f"  Warning: No candidates found for pair {i}, skipping.")
            continue
        
        # Debug: check first pair's coordinates
        if i == 0:
            print(f"\n  DEBUG pair 0:")
            print(f"    GT: ({gt_x}, {gt_y})")
            print(f"    Search image shape: {search.shape}")
            print(f"    Reference patch shape: {reference_ncc.shape}")
            print(f"    Top 3 candidates:")
            for idx, (cx, cy, sc) in enumerate(candidates[:3]):
                dist = np.sqrt((cx - gt_x)**2 + (cy - gt_y)**2)
                print(f"      [{idx}] ({cx:.1f}, {cy:.1f}) score={sc:.4f} dist_to_gt={dist:.1f}px")
        
        # For each candidate, crop patch and label it
        for candidate_idx, (cand_x, cand_y, score) in enumerate(candidates):
            # Crop patch from search image at candidate location
            # Use a slightly larger patch size for training (to have enough context)
            patch_cand = crop_patch_at_location(search, cand_x, cand_y, patch_size=ref_patch_size)
            
            # Use the reference_ncc patch as the reference for training
            patch_ref = reference_ncc
            
            # Label candidate: ground truth and candidates should be in same coordinate system
            label = label_candidate(cand_x, cand_y, gt_x, gt_y, threshold_px=5.0)
            
            all_patches_ref.append(patch_ref)
            all_patches_cand.append(patch_cand)
            all_labels.append(label)
            label_counts["positive" if label == 1 else "negative"] += 1
            
            all_metadata.append({
                "pair_id": i,
                "candidate_idx": candidate_idx,
                "candidate_x": float(cand_x),
                "candidate_y": float(cand_y),
                "ncc_score": float(score),
                "gt_x": float(gt_x),
                "gt_y": float(gt_y),
                "label": int(label),
                "add_marker": pair_meta["add_marker"],
                "architecture": pair_meta["architecture"]
            })
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_pairs} pairs, "
                  f"{len(all_patches_ref)} candidate patches so far...")
    
    print(f"\nTotal candidate patches generated: {len(all_patches_ref)}")
    print(f"  Positive (label=1): {label_counts['positive']}")
    print(f"  Negative (label=0): {label_counts['negative']}")
    if label_counts['positive'] + label_counts['negative'] > 0:
        pos_pct = 100.0 * label_counts['positive'] / (label_counts['positive'] + label_counts['negative'])
        print(f"  Class balance: {pos_pct:.1f}% positive, {100-pos_pct:.1f}% negative")
    
    # Analyze distance distribution
    distances = []
    for meta in all_metadata:
        dist = np.sqrt((meta['candidate_x'] - meta['gt_x'])**2 + (meta['candidate_y'] - meta['gt_y'])**2)
        distances.append(dist)
    
    distances = np.array(distances)
    print(f"\nDistance distribution (candidate to GT):")
    print(f"  Min: {distances.min():.2f}px, Max: {distances.max():.2f}px, Mean: {distances.mean():.2f}px")
    print(f"  Std: {distances.std():.2f}px, Median: {np.median(distances):.2f}px")
    print(f"  Within 5px: {(distances <= 5).sum()}, Within 10px: {(distances <= 10).sum()}")
    print(f"  Within 20px: {(distances <= 20).sum()}, Within 50px: {(distances <= 50).sum()}")
    
    # Convert to numpy arrays
    patches_ref_array = np.array(all_patches_ref, dtype=np.uint8)
    patches_cand_array = np.array(all_patches_cand, dtype=np.uint8)
    labels_array = np.array(all_labels, dtype=np.int32)
    
    print(f"\nReference patches shape: {patches_ref_array.shape}")
    print(f"Candidate patches shape: {patches_cand_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    
    # Split into train/val/test
    num_samples = len(all_patches_ref)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(num_samples * train_split)
    val_end = train_end + int(num_samples * val_split)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    patches_ref_train = patches_ref_array[train_idx]
    patches_cand_train = patches_cand_array[train_idx]
    labels_train = labels_array[train_idx]
    
    patches_ref_val = patches_ref_array[val_idx]
    patches_cand_val = patches_cand_array[val_idx]
    labels_val = labels_array[val_idx]
    
    patches_ref_test = patches_ref_array[test_idx]
    patches_cand_test = patches_cand_array[test_idx]
    labels_test = labels_array[test_idx]
    
    print(f"\nTrain/val/test split:")
    print(f"  Train: {len(train_idx)} samples ({100*len(train_idx)/num_samples:.1f}%)")
    print(f"    Positive: {(labels_train == 1).sum()}, Negative: {(labels_train == 0).sum()}")
    print(f"  Val:   {len(val_idx)} samples ({100*len(val_idx)/num_samples:.1f}%)")
    print(f"    Positive: {(labels_val == 1).sum()}, Negative: {(labels_val == 0).sum()}")
    print(f"  Test:  {len(test_idx)} samples ({100*len(test_idx)/num_samples:.1f}%)")
    print(f"    Positive: {(labels_test == 1).sum()}, Negative: {(labels_test == 0).sum()}")
    
    # Save to .npz
    output_file = os.path.join(os.path.dirname(__file__), "training_data.npz")
    np.savez_compressed(
        output_file,
        patches_ref_train=patches_ref_train,
        patches_cand_train=patches_cand_train,
        labels_train=labels_train,
        patches_ref_val=patches_ref_val,
        patches_cand_val=patches_cand_val,
        labels_val=labels_val,
        patches_ref_test=patches_ref_test,
        patches_cand_test=patches_cand_test,
        labels_test=labels_test
    )
    print(f"\nSaved to: {output_file}")
    
    # Save metadata as JSON separately
    metadata_file = os.path.join(os.path.dirname(__file__), "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal generation time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    return {
        "patches_ref_train": patches_ref_train,
        "patches_cand_train": patches_cand_train,
        "labels_train": labels_train,
        "patches_ref_val": patches_ref_val,
        "patches_cand_val": patches_cand_val,
        "labels_val": labels_val,
        "patches_ref_test": patches_ref_test,
        "patches_cand_test": patches_cand_test,
        "labels_test": labels_test,
        "metadata": all_metadata,
        "time_seconds": elapsed,
        "label_counts": label_counts,
        "class_balance": pos_pct if (label_counts['positive'] + label_counts['negative'] > 0) else 0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=500,
                       help="Number of pairs to generate")
    parser.add_argument("--output_dir", default="../tests/synthetic_data",
                       help="Output directory for training pairs")
    parser.add_argument("--ref_patch_size", type=int, default=128,
                       help="Size of reference patch for NCC matching")
    args = parser.parse_args()
    
    data = generate_training_data(
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        ref_patch_size=args.ref_patch_size
    )
    
    print("\n" + "="*70)
    print("CHECKPOINT: Training data generation complete")
    print("="*70)
    print("\nSummary:")
    print(f"  Total patches: {len(data['metadata'])}")
    print(f"  Positive samples: {data['label_counts']['positive']}")
    print(f"  Negative samples: {data['label_counts']['negative']}")
    print(f"  Class balance: {data['class_balance']:.1f}% positive")
    print(f"  Train set: {len(data['labels_train'])} samples")
    print(f"  Val set:   {len(data['labels_val'])} samples")
    print(f"  Test set:  {len(data['labels_test'])} samples")
    print(f"  Time taken: {data['time_seconds']:.1f}s ({data['time_seconds']/60:.1f}m)")
    print("\n" + "="*70)
