import csv

gt_rows = list(csv.DictReader(open("../tests/synthetic_data/ground_truth.csv")))
results = list(csv.DictReader(open("batch_results.csv")))

hard_ids = {r["pair_id"] for r in gt_rows if r["add_marker"] == "False"}

print(f"{'pair_id':>8} {'arch':>8} {'pixel_error':>12}")
for r in results:
    if r["pair_id"] in hard_ids:
        print(f"{r['pair_id']:>8} {r['architecture']:>8} {float(r['pixel_error']):>12.2f}")