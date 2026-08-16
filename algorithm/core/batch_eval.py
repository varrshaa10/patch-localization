import csv
import time
import numpy as np
from infer import run_inference

results = []
with open("../tests/synthetic_data/ground_truth.csv") as f:
    rows = list(csv.DictReader(f))

print(f"{'ID':<4}{'Arch':<8}{'Mark':<6}{'GT (x,y)':<16}{'Pred (x,y)':<16}{'Error':>9}  {'Conf':<6}")
print("-" * 80)

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
    dx = pred_x - gt_x
    dy = pred_y - gt_y

    results.append({
        "pair_id": pair_id,
        "architecture": row["architecture"],
        "add_marker": add_marker,
        "gt_x": gt_x, "gt_y": gt_y,
        "pred_x": pred_x, "pred_y": pred_y,
        "dx": dx,
        "dy": dy,
        "pixel_error": error,
        "ncc_score": score,
        "ambiguity_ratio": ratio,
        "confidence": confidence,
        "time_sec": elapsed
    })

    gt_coord = f"({gt_x:.0f},{gt_y:.0f})"
    pred_coord = f"({pred_x},{pred_y})"
    print(f"{pair_id:<4}{row['architecture']:<8}{str(add_marker):<6}"
          f"{gt_coord:<16}{pred_coord:<16}{error:>7.2f}px  {confidence:<6}")

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