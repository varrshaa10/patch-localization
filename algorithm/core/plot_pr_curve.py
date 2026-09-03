import csv
import matplotlib.pyplot as plt

INPUT_CSV = "batch_results.csv"
OUTPUT_PNG = "pr_curve.png"
THRESHOLDS = list(range(0, 51))  # 0 to 50 px

def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "architecture": row["architecture"],
                "pixel_error": float(row["pixel_error"]),
            })
    return rows

def accuracy_at_thresholds(rows, thresholds):
    n = len(rows)
    if n == 0:
        return [0 for _ in thresholds]
    accs = []
    for t in thresholds:
        correct = sum(1 for r in rows if r["pixel_error"] <= t)
        accs.append(correct / n * 100)
    return accs

def main():
    rows = load_results(INPUT_CSV)
    all_accs = accuracy_at_thresholds(rows, THRESHOLDS)

    dram_rows = [r for r in rows if r["architecture"] == "dram"]
    finfet_rows = [r for r in rows if r["architecture"] == "finfet"]
    dram_accs = accuracy_at_thresholds(dram_rows, THRESHOLDS)
    finfet_accs = accuracy_at_thresholds(finfet_rows, THRESHOLDS)

    plt.figure(figsize=(8, 6))
    plt.plot(THRESHOLDS, all_accs, label="Overall", linewidth=2, color="black")
    plt.plot(THRESHOLDS, dram_accs, label="DRAM", linestyle="--")
    plt.plot(THRESHOLDS, finfet_accs, label="FinFET", linestyle="--")
    plt.xlabel("Pixel-error tolerance (px)")
    plt.ylabel("Accuracy (%)")
    plt.title("Localization Accuracy vs Pixel-Tolerance Threshold")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print("Saved plot to " + OUTPUT_PNG)
    print(f"Overall accuracy at 1px: {all_accs[1]:.1f}%")
    print(f"Overall accuracy at 5px: {all_accs[5]:.1f}%")
    print(f"Overall accuracy at 50px: {all_accs[50]:.1f}%")

if __name__ == "__main__":
    main()
