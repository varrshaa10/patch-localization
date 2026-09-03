"""Score a generated Phase 2 synthetic localization dataset."""
import argparse
import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH = ROOT / "algorithm" / "tests" / "synthetic_data_phase2_v2" / "ground_truth_combined.csv"
DEFAULT_PREDICTIONS = ROOT / "outputs" / "final_main_predictions_v2.csv"
DEFAULT_TIMINGS = ROOT / "outputs" / "final_main_predictions_v2_timings.csv"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def credit_from_error(error):
    if error <= 1.0:
        return 1.00
    if error <= 2.0:
        return 0.80
    if error <= 3.0:
        return 0.60
    if error <= 5.0:
        return 0.40
    return 0.00


def evaluate(ground_truth_path, predictions_path, timings_path):
    ground_truth = {row["pair_id"]: row for row in read_rows(ground_truth_path)}
    predictions = {row["pair_id"]: row for row in read_rows(predictions_path)}
    timing_rows = read_rows(timings_path)
    missing = sorted(set(ground_truth) - set(predictions))
    if missing:
        raise ValueError(f"Missing prediction rows: {', '.join(missing[:5])}")
    if set(ground_truth) != {row["pair_id"] for row in timing_rows}:
        raise ValueError("Timing rows do not match ground-truth rows")

    tp = fp = fn = tn = 0
    present_errors = []
    present_credits = []
    within_5 = 0
    category_stats = {}
    scale_errors = []
    theta_errors = []

    for pair_id, gt_row in ground_truth.items():
        prediction = predictions[pair_id]
        category = gt_row["category"]
        found = int(float(prediction.get("found", 0) or 0)) == 1
        present = category != "absent"
        stats = category_stats.setdefault(category, {"pairs": 0, "found": 0, "credits": [], "within_5": 0})
        stats["pairs"] += 1
        stats["found"] += int(found)

        if present:
            if not found:
                fn += 1
                stats["credits"].append(0.0)
                continue
            tp += 1
            error = math.hypot(
                float(prediction.get("x", 0) or 0) - float(gt_row["gt_x"]),
                float(prediction.get("y", 0) or 0) - float(gt_row["gt_y"]),
            )
            credit = credit_from_error(error)
            present_errors.append(error)
            present_credits.append(credit)
            stats["credits"].append(credit)
            if error <= 5.0:
                within_5 += 1
                stats["within_5"] += 1
            if credit > 0.0:
                scale_errors.append(abs(float(prediction.get("scale", 0) or 0) - float(gt_row["gt_scale"])) / float(gt_row["gt_scale"]))
                theta_errors.append(abs(float(prediction.get("theta", 0) or 0) - float(gt_row["gt_theta"])))
        else:
            if found:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    runtimes = [float(row["time_sec"]) for row in timing_rows]

    print("Synthetic Phase 2 scoring report")
    print("================================")
    print(f"Predictions file: {predictions_path}")
    print(f"Ground truth file: {ground_truth_path}")
    print(f"Timings file: {timings_path}")
    print()
    print(f"Total pairs: {len(ground_truth)}")
    print(f"Present pairs: {tp + fn}")
    print(f"Absent pairs: {fp + tn}")
    print()
    print("Category results:")
    for category, stats in category_stats.items():
        mean_credit = statistics.mean(stats["credits"]) if stats["credits"] else 0.0
        print(f"  {category}: {stats['pairs']} pairs, found={stats['found']}, mean credit={mean_credit:.3f}, within_5px={stats['within_5']}")
    print()
    print("Rejection metrics:")
    print(f"  precision: {precision:.6f}")
    print(f"  recall:    {recall:.6f}")
    print(f"  F1:        {f1:.6f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print()
    print("Localization and pose:")
    print(f"  within 5 px: {within_5}/{tp + fn}")
    print(f"  mean present credit: {statistics.mean(present_credits) if present_credits else 0.0:.6f}")
    print(f"  median pixel error: {statistics.median(present_errors) if present_errors else 'N/A'}")
    print(f"  scale error median: {statistics.median(scale_errors) * 100 if scale_errors else 0.0:.3f}%")
    print(f"  theta error median: {statistics.median(theta_errors) if theta_errors else 0.0:.3f} degrees")
    print()
    print("Runtime:")
    print(f"  median time_sec: {statistics.median(runtimes):.6f}s")
    print(f"  max time_sec:    {max(runtimes):.6f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground_truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--timings", type=Path, default=DEFAULT_TIMINGS)
    args = parser.parse_args()
    evaluate(args.ground_truth, args.predictions, args.timings)


if __name__ == "__main__":
    main()
