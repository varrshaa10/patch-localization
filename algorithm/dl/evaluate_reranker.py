"""End-to-end reranker evaluation script for the synthetic dataset.

This is an experiment-only script and is intentionally separate from the classical
NCC pipeline. It compares the baseline NCC candidate selection and the reranker on
held-out synthetic pairs, reporting mean/median error and tolerance accuracy.
"""

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "algorithm" / "core"
DL_DIR = ROOT / "algorithm" / "dl"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from model import SiameseCNN
from ncc import ncc_match_multi

DATA_DIR = ROOT / "algorithm" / "tests" / "synthetic_data"
CSV_PATH = DATA_DIR / "ground_truth_combined.csv"
MODEL_PATH = DL_DIR / "reranker_model.pt"
PATCH_SIZE = 128


def crop_center_patch(image, cx, cy, patch_size=PATCH_SIZE):
    h, w = image.shape[:2]
    x0 = max(0, int(cx - patch_size // 2))
    y0 = max(0, int(cy - patch_size // 2))
    x1 = min(w, int(cx + patch_size // 2))
    y1 = min(h, int(cy + patch_size // 2))
    patch = image[y0:y1, x0:x1]
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded = np.ones((patch_size, patch_size), dtype=image.dtype) * 30
        py0 = max(0, int(patch_size // 2 - cy))
        px0 = max(0, int(patch_size // 2 - cx))
        padded[py0:py0 + patch.shape[0], px0:px0 + patch.shape[1]] = patch
        patch = padded
    return patch


def pick_plain(candidates, search):
    if not candidates:
        return None
    best_score = max(c[2] for c in candidates)
    near_ties = [c for c in candidates if c[2] >= best_score - 0.005]
    h, w = search.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    return min(near_ties, key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - center))


def rerank_candidates(model, search, reference, candidates):
    if not candidates:
        return None
    ref_patch = crop_center_patch(reference, float(reference.shape[1]) / 2.0, float(reference.shape[0]) / 2.0, PATCH_SIZE)
    ref_tensor = torch.from_numpy(ref_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    scored = []
    for cx, cy, score in candidates:
        cand_patch = crop_center_patch(search, cx, cy, PATCH_SIZE)
        cand_tensor = torch.from_numpy(cand_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(ref_tensor, cand_tensor).item()
        scored.append((cx, cy, score, pred))
    return max(scored, key=lambda x: x[3])


def summarize(errors):
    arr = np.array(errors, dtype=float)
    summary = {"n": len(arr), "mean": float(arr.mean()) if len(arr) else float("nan"), "median": float(np.median(arr)) if len(arr) else float("nan")}
    for tol in [1, 5, 10, 50]:
        summary[f"acc_{tol}px"] = float((arr <= tol).mean()) if len(arr) else float("nan")
    return summary


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = SiameseCNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = list(csv.DictReader(CSV_PATH.open(newline="")))
    marker_rows = [r for r in rows if str(r.get("add_marker", "")).strip().lower() in ("true", "1", "yes", "y")]
    no_marker_rows = [r for r in rows if str(r.get("add_marker", "")).strip().lower() not in ("true", "1", "yes", "y")]

    def evaluate_split(name, subset):
        plain_errors = []
        rerank_errors = []

        for row in subset:
            pair_id = int(row["pair_id"])
            ref = cv2.imread(str((DATA_DIR / f"pair_{pair_id}" / "reference.png").resolve()), cv2.IMREAD_GRAYSCALE)
            search = cv2.imread(str((DATA_DIR / f"pair_{pair_id}" / "search.png").resolve()), cv2.IMREAD_GRAYSCALE)
            if ref is None or search is None:
                continue

            gt_x = float(row["gt_x"])
            gt_y = float(row["gt_y"])

            ref_patch = crop_center_patch(search, gt_x, gt_y, PATCH_SIZE)
            candidates = ncc_match_multi(search, ref_patch, num_peaks=8, min_distance=15)
            if not candidates:
                continue

            plain = pick_plain(candidates, search)
            if plain is not None:
                px, py, _ = plain
                plain_errors.append(math.hypot(px - gt_x, py - gt_y))

            reranked = rerank_candidates(model, search, ref, candidates)
            if reranked is not None:
                px, py, _, _ = reranked
                rerank_errors.append(math.hypot(px - gt_x, py - gt_y))

        print(f"\n=== {name} ===")
        plain_summary = summarize(plain_errors)
        rerank_summary = summarize(rerank_errors)
        print(f"Plain NCC:  n={plain_summary['n']}, mean={plain_summary['mean']:.2f}px, median={plain_summary['median']:.2f}px, acc_1={plain_summary['acc_1px']:.3f}, acc_5={plain_summary['acc_5px']:.3f}, acc_10={plain_summary['acc_10px']:.3f}, acc_50={plain_summary['acc_50px']:.3f}")
        print(f"Reranked:   n={rerank_summary['n']}, mean={rerank_summary['mean']:.2f}px, median={rerank_summary['median']:.2f}px, acc_1={rerank_summary['acc_1px']:.3f}, acc_5={rerank_summary['acc_5px']:.3f}, acc_10={rerank_summary['acc_10px']:.3f}, acc_50={rerank_summary['acc_50px']:.3f}")

    evaluate_split("MARKER", marker_rows)
    evaluate_split("NO_MARKER", no_marker_rows)
    evaluate_split("ALL", rows)


if __name__ == "__main__":
    main()
