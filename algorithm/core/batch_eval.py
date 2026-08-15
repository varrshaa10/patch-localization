import csv
import time
import numpy as np
from infer import run_inference

results = []
with open("../tests/synthetic_data/ground_truth.csv") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    pair_id = row["pair_id"]
    ref_path = row["ref_path"]
    search_path = row["search_path"]
    gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
    add_marker = row.get("add_marker", "")

    start = time.time()
    pred_x, pred_y, score, ratio, confidence, angle, scale, candidates = run_inference(ref_path, search_path)
    elapsed = time.time() - start

    error = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
    results.append({
        "pair_id": pair_id,
        "architecture": row["architecture"],
        "add_marker": add_marker,
        "gt_x": gt_x, "gt_y": gt_y,
        "pred_x": pred_x, "pred_y": pred_y,
        "pixel_error": error,
        "ncc_score": score,
        "ambiguity_ratio": ratio,
        "confidence": confidence,
        "time_sec": elapsed
    })
    print(f"Pair {pair_id} ({row['architecture']}, marker={add_marker}): "
          f"error={error:.2f}px, score={score:.3f}, ratio={ratio:.3f}, "
          f"confidence={confidence}, time={elapsed:.2f}s")

errors = np.array([r["pixel_error"] for r in results])
times = np.array([r["time_sec"] for r in results])
ratios = np.array([r["ambiguity_ratio"] for r in results])

print("\n--- Summary ---")
print(f"Total pairs: {len(results)}")
print(f"Mean pixel error: {errors.mean():.2f}")
print(f"Median pixel error: {np.median(errors):.2f}")
print(f"Max pixel error: {errors.max():.2f}")
print(f"Mean inference time: {times.mean():.2f}s")
for tol in [1, 2, 3, 5, 10, 20, 50]:
    pct = (errors <= tol).mean() * 100
    print(f"Accuracy within {tol}px: {pct:.1f}%")

# --- Ratio distribution, split by whether the pair actually failed ---
print("\n--- Ambiguity ratio distribution ---")
failed_mask = errors > 50   # your existing "hard case" cutoff
print(f"Ratio on FAILED pairs (n={failed_mask.sum()}): "
      f"min={ratios[failed_mask].min():.3f}, max={ratios[failed_mask].max():.3f}, "
      f"mean={ratios[failed_mask].mean():.3f}")
print(f"Ratio on PASSED pairs (n={(~failed_mask).sum()}): "
      f"min={ratios[~failed_mask].min():.3f}, max={ratios[~failed_mask].max():.3f}, "
      f"mean={ratios[~failed_mask].mean():.3f}")

with open("batch_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print("\nSaved batch_results.csv")