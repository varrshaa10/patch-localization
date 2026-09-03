import csv
import math
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = (ROOT / "algorithm" / "tests" / "synthetic_data_phase2").resolve()
sys.path.insert(0, str(ROOT / "algorithm" / "core"))
import infer as inf


def _normalize_pair_id(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    return str(int(float(text)))


def load_ground_truth_rows():
    gt_path = DATASET_ROOT / "ground_truth_combined.csv"
    with gt_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    normalized = []
    for row in rows:
        pair_id = _normalize_pair_id(row.get("pair_id"))
        if pair_id == "":
            continue
        gt_x_raw = row.get("gt_x", "")
        gt_y_raw = row.get("gt_y", "")
        if gt_x_raw in (None, "") or gt_y_raw in (None, ""):
            continue
        normalized.append({
            **row,
            "pair_id": pair_id,
            "gt_x": float(gt_x_raw),
            "gt_y": float(gt_y_raw),
            "found": int(float(row.get("found", 0) or 0)),
        })
    return normalized


root = DATASET_ROOT
gt_rows = load_ground_truth_rows()
present = [r for r in gt_rows if r["found"] == 1]


def pair_files(pair_id):
    pid = _normalize_pair_id(pair_id)
    return root / f"pair_{pid}" / "reference.png", root / f"pair_{pid}" / "search.png"


def produce_candidates(search, ref):
    angles = inf._grid_values(-inf.ANGLE_RANGE, inf.ANGLE_RANGE, inf.ANGLE_STEP)
    scales = inf._grid_values(
        inf.SCALE_CENTER - inf.SCALE_RANGE,
        inf.SCALE_CENTER + inf.SCALE_RANGE,
        inf.SCALE_STEP,
    )
    out = []
    for angle in angles:
        for scale in scales:
            scaled = inf._prepare_template_variant(ref, angle, scale)
            if scaled.shape[0] < 4 or scaled.shape[1] < 4:
                continue
            if scaled.shape[0] >= search.shape[0] or scaled.shape[1] >= search.shape[1]:
                continue
            result = cv2.matchTemplate(search, scaled, cv2.TM_CCOEFF_NORMED)
            rc = result.copy()
            for _ in range(8):
                _, max_val, _, max_loc = cv2.minMaxLoc(rc)
                x0, y0 = max_loc
                h, w = scaled.shape[:2]
                x = x0 + w // 2
                y = y0 + h // 2
                out.append((x, y, float(max_val), float(angle), float(scale)))
                yy0 = max(0, y0 - 15)
                yy1 = min(rc.shape[0], y0 + 15)
                xx0 = max(0, x0 - 15)
                xx1 = min(rc.shape[1], x0 + 15)
                rc[yy0:yy1, xx0:xx1] = -1
    out.sort(key=lambda item: item[2], reverse=True)
    return out[:20]

base_hits = 0
weighted_hits = 0
phase_hits = 0
consistency_hits = 0
ensemble_hits = 0
weighted_times = []
phase_times = []
consistency_times = []
ensemble_times = []
weighted_changed = []
phase_changed = []
consistency_changed = []
ensemble_changed = []


def compare_debug_samples(pairs_to_check):
    for pair_id in pairs_to_check:
        row = next(r for r in present if str(r["pair_id"]) == str(pair_id))
        ref_path, search_path = pair_files(pair_id)
        ref = cv2.imread(str(ref_path))
        search = cv2.imread(str(search_path))
        if ref is None or search is None:
            raise FileNotFoundError(pair_id)
        gt_x = float(row["gt_x"]) 
        gt_y = float(row["gt_y"])
        benchmark_best, _, _, _, _ = inf.coarse_to_fine_search(search, ref)
        benchmark_pred = (benchmark_best[0], benchmark_best[1])
        infer_pred = inf.run_inference(str(ref_path), str(search_path))[:2]
        print(
            f"pair_id={pair_id}: benchmark={(benchmark_pred[0], benchmark_pred[1])} gt={(gt_x, gt_y)} infer={infer_pred} "
            f"benchmark_err={math.hypot(benchmark_pred[0] - gt_x, benchmark_pred[1] - gt_y):.3f} "
            f"infer_err={math.hypot(infer_pred[0] - gt_x, infer_pred[1] - gt_y):.3f}"
        )


# Explicitly compare the benchmark path against the same data-flow used by register.py/infer.py.
# This is the check that exposes whether the benchmark is accidentally evaluating a different search
# function or a stale/mismatched CSV file path.
compare_debug_samples([0, 1, 2])

for row in present:
    pair_id = str(row["pair_id"])
    ref_path, search_path = pair_files(pair_id)
    ref = cv2.imread(str(ref_path))
    search = cv2.imread(str(search_path))
    if ref is None or search is None:
        raise FileNotFoundError(pair_id)

    gt_x = float(row["gt_x"])
    gt_y = float(row["gt_y"])
    raw_best, _, _, _, _ = inf.coarse_to_fine_search(search, ref)
    raw_err = math.hypot(raw_best[0] - gt_x, raw_best[1] - gt_y)
    if raw_err <= 5:
        base_hits += 1

    cand = produce_candidates(search, ref)
    for label, reranker in [
        ("weighted", inf.rerank_peak_candidates_weighted_ncc),
        ("phase", inf.rerank_peak_candidates_phase_correlation),
        ("consistency", inf.rerank_peak_candidates_subregion_consistency),
    ]:
        t0 = time.perf_counter()
        rer = reranker(cand, search, ref)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if label == "weighted":
            weighted_times.append(elapsed_ms)
        elif label == "phase":
            phase_times.append(elapsed_ms)
        else:
            consistency_times.append(elapsed_ms)

        if not rer:
            continue

        x, y, _, _, _ = rer[0]
        err = math.hypot(x - gt_x, y - gt_y)
        if label == "weighted":
            if err <= 5:
                weighted_hits += 1
            if (raw_err > 5 and err <= 5) or (raw_err <= 5 and err > 5):
                weighted_changed.append(pair_id)
        elif label == "phase":
            if err <= 5:
                phase_hits += 1
            if (raw_err > 5 and err <= 5) or (raw_err <= 5 and err > 5):
                phase_changed.append(pair_id)
        else:
            if err <= 5:
                consistency_hits += 1
            if (raw_err > 5 and err <= 5) or (raw_err <= 5 and err > 5):
                consistency_changed.append(pair_id)

    t0 = time.perf_counter()
    ensemble = inf.majority_vote_ensemble(search, ref, candidate_pool=cand)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ensemble_times.append(elapsed_ms)
    if ensemble is not None:
        x, y, _, _, _ = ensemble
        err = math.hypot(x - gt_x, y - gt_y)
        if err <= 5:
            ensemble_hits += 1
        if (raw_err > 5 and err <= 5) or (raw_err <= 5 and err > 5):
            ensemble_changed.append(pair_id)

print(f"BASELINE_RAW_WITHIN5 {base_hits} / {len(present)}")
print(f"WEIGHTED_WITHIN5 {weighted_hits} / {len(present)}")
print(f"PHASE_WITHIN5 {phase_hits} / {len(present)}")
print(f"CONSISTENCY_WITHIN5 {consistency_hits} / {len(present)}")
print(f"ENSEMBLE_WITHIN5 {ensemble_hits} / {len(present)}")
print(f"WEIGHTED_EXTRA_MS_PER_PAIR {sum(weighted_times) / len(weighted_times):.3f}")
print(f"PHASE_EXTRA_MS_PER_PAIR {sum(phase_times) / len(phase_times):.3f}")
print(f"CONSISTENCY_EXTRA_MS_PER_PAIR {sum(consistency_times) / len(consistency_times):.3f}")
print(f"ENSEMBLE_EXTRA_MS_PER_PAIR {sum(ensemble_times) / len(ensemble_times):.3f}")
print("WEIGHTED_CHANGED", weighted_changed)
print("PHASE_CHANGED", phase_changed)
print("CONSISTENCY_CHANGED", consistency_changed)
print("ENSEMBLE_CHANGED", ensemble_changed)
