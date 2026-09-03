"""Compare registration scores for present and absent official pairs."""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TIMINGS = ROOT / "algorithm" / "tests" / "official_phase2" / "predictions_timings.csv"
DEFAULT_GROUND_TRUTH = ROOT / "algorithm" / "tests" / "official_phase2" / "ground_truth.csv"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, default=DEFAULT_TIMINGS)
    parser.add_argument("--ground_truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    args = parser.parse_args()

    timings = {row["pair_id"]: row for row in read_rows(args.timings)}
    ground_truth = {row["pair_id"]: row for row in read_rows(args.ground_truth)}
    joined = []
    for pair_id, gt_row in ground_truth.items():
        timing = timings.get(pair_id)
        if timing is None:
            raise KeyError(f"Missing timing row for {pair_id}")
        joined.append({
            "pair_id": pair_id,
            "label": "GT_PRESENT" if int(float(gt_row["present"])) else "GT_ABSENT",
            "best_score": float(timing["best_score"]),
            "top2_margin": float(timing["top2_margin"]),
            "found": timing.get("found", ""),
        })

    for label in ("GT_PRESENT", "GT_ABSENT"):
        label_rows = [item for item in joined if item["label"] == label]
        for metric in ("best_score", "top2_margin"):
            print(f"{label} {metric} (sorted):")
            for row in sorted(label_rows, key=lambda item: item[metric]):
                print(f"  {row['pair_id']}: {row[metric]:.6f} found={row['found']}")
            print()

    present_scores = [row["best_score"] for row in joined if row["label"] == "GT_PRESENT"]
    absent_scores = [row["best_score"] for row in joined if row["label"] == "GT_ABSENT"]
    print(f"min best_score among GT_PRESENT: {min(present_scores):.6f}")
    print(f"max best_score among GT_ABSENT: {max(absent_scores):.6f}")


if __name__ == "__main__":
    main()
