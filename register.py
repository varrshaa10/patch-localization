#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE_DIR = ROOT / "algorithm" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from infer import (  # noqa: E402
    _candidate_clusters,
    rerank_peak_candidates_phase_correlation,
)
from ncc import ncc_match, ncc_match_multi  # noqa: E402
from rotation_search import rotate_image  # noqa: E402
from scale_rotation_search import scale_template  # noqa: E402

OUTPUT_FIELDS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
TIME_LOG_FIELDS = ["pair_id", "time_sec", "best_score", "top2_margin", "found", "status", "error"]
TIMEOUT_SECONDS = 15.0
FIND_THRESHOLD = 0.0002
SCORE_THRESHOLD = 0.30
TRACE_ERRORS = os.environ.get("REGISTER_TRACE_ERRORS", "1") == "1"


def _resolve_csv_path(raw_value, csv_dir):
    if raw_value is None:
        return ""
    value = str(raw_value).strip()
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (csv_dir / path).resolve()
    return str(path)


def _canonical_value(row, *candidates):
    if row is None:
        return None
    normalized = {str(k).strip(): v for k, v in row.items()}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _zero_result(pair_id):
    return {
        "pair_id": str(pair_id),
        "x": 0,
        "y": 0,
        "theta": 0,
        "scale": 0,
        "found": 0,
        "score": 0.0,
    }


def _two_stage_search(image, template):
    """Rank the full pose grid cheaply, then search only its best five poses."""
    import cv2
    import numpy as np

    image_small = cv2.resize(
        image, (max(1, image.shape[1] // 2), max(1, image.shape[0] // 2)),
        interpolation=cv2.INTER_AREA,
    )
    template_small = cv2.resize(
        template, (max(1, template.shape[1] // 2), max(1, template.shape[0] // 2)),
        interpolation=cv2.INTER_AREA,
    )

    angle_range = 5
    angle_step = 1
    zoom_min = 8.0
    zoom_max = 12.0
    zoom_step = 0.5
    pose_scores = []
    for angle in np.arange(-angle_range, angle_range + angle_step, angle_step):
        rotated = rotate_image(template_small, angle)
        for zoom in np.arange(zoom_min, zoom_max + zoom_step, zoom_step):
            scale = 1.0 / zoom
            scaled = scale_template(rotated, scale)
            if scaled.shape[0] < 4 or scaled.shape[1] < 4:
                continue
            if scaled.shape[0] >= image_small.shape[0] or scaled.shape[1] >= image_small.shape[1]:
                continue
            _, _, score = ncc_match(image_small, scaled)
            pose_scores.append((float(score), float(angle), float(scale)))

    if not pose_scores:
        raise RuntimeError("No valid coarse candidates found -- check image/template sizes.")

    all_candidates = []
    for _, angle, scale in sorted(pose_scores, reverse=True)[:5]:
        rotated = rotate_image(template, angle)
        scaled = scale_template(rotated, scale)
        if scaled.shape[0] < 4 or scaled.shape[1] < 4:
            continue
        if scaled.shape[0] >= image.shape[0] or scaled.shape[1] >= image.shape[1]:
            continue
        peaks = ncc_match_multi(image, scaled, num_peaks=6, min_distance=15)
        all_candidates.extend((x, y, score, angle, scale) for x, y, score in peaks)

    if not all_candidates:
        raise RuntimeError("No valid full-resolution candidates found.")

    img_h, img_w = image.shape[:2]
    img_center = np.array([img_w / 2, img_h / 2])
    best_score = max(candidate[2] for candidate in all_candidates)
    near_best = [candidate for candidate in all_candidates if candidate[2] >= best_score - 0.005]
    best = min(
        near_best,
        key=lambda candidate: np.linalg.norm(
            np.array([candidate[0], candidate[1]]) - img_center
        ),
    )
    sorted_clusters, top_candidates, top2_margin = _candidate_clusters(all_candidates, cluster_dist=20)
    ratio = 0.0 if len(sorted_clusters) < 2 else sorted_clusters[1][1] / sorted_clusters[0][1]
    return best, ratio, top_candidates, sorted_clusters, top2_margin


def _worker(pair_id, reference_path, search_path, queue):
    try:
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"reference image missing: {reference_path}")
        if not os.path.exists(search_path):
            raise FileNotFoundError(f"search image missing: {search_path}")

        import cv2
        image = cv2.imread(search_path)
        template = cv2.imread(reference_path)
        if image is None:
            raise FileNotFoundError(f"Could not read search image: {search_path}")
        if template is None:
            raise FileNotFoundError(f"Could not read reference image: {reference_path}")

        best, _, top_candidates, _, top2_margin = _two_stage_search(image, template)
        if top_candidates:
            phase_candidates = [
                (float(x), float(y), float(score), 0.0, 0.0)
                for x, y, score in top_candidates
            ]
            phase_ranked = rerank_peak_candidates_phase_correlation(phase_candidates, image, template)
            if phase_ranked:
                reranked_x, reranked_y, reranked_score, _, _ = phase_ranked[0]
                best = (reranked_x, reranked_y, reranked_score, best[3], best[4])
        x, y, best_score, theta, internal_scale = best
        margin = _safe_float(top2_margin, 0.0)
        if not math.isfinite(margin):
            margin = 1.0
        output_scale = 0.0 if internal_scale in (None, 0) else (1.0 / float(internal_scale))

        if margin > FIND_THRESHOLD and best_score > SCORE_THRESHOLD:
            score_value = best_score
            pose = {
                "pair_id": str(pair_id),
                "x": int(round(float(x))),
                "y": int(round(float(y))),
                "theta": float(theta),
                "scale": float(output_scale),
                "found": 1,
                "score": score_value,
            }
        else:
            score_value = best_score
            pose = {
                "pair_id": str(pair_id),
                "x": 0,
                "y": 0,
                "theta": 0,
                "scale": 0,
                "found": 0,
                "score": score_value,
            }
        queue.put(("ok", pose, margin))
    except BaseException as exc:
        if TRACE_ERRORS:
            traceback.print_exc()
        queue.put(("error", _zero_result(pair_id), 0.0, f"{type(exc).__name__}: {exc}"))


def _process_pair(pair_id, reference_path, search_path):
    queue = Queue()
    process = Process(target=_worker, args=(pair_id, reference_path, search_path, queue), daemon=True)
    start = time.perf_counter()
    process.start()
    process.join(timeout=TIMEOUT_SECONDS)
    elapsed = time.perf_counter() - start

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"pair_id={pair_id}: timeout after {TIMEOUT_SECONDS}s", file=sys.stderr)
        return _zero_result(pair_id), elapsed, 0.0, "timeout"

    try:
        status, payload, margin, *rest = queue.get_nowait()
    except Exception:
        print(f"pair_id={pair_id}: no worker result; process exited unexpectedly", file=sys.stderr)
        return _zero_result(pair_id), elapsed, 0.0, "empty"

    if status == "ok":
        print(
            f"pair_id={pair_id} best_score={payload['score']:.6f} top2_margin={margin:.6f}",
            file=sys.stderr,
        )
        return payload, elapsed, margin, None

    error_message = rest[0] if rest else "unknown error"
    print(f"pair_id={pair_id}: {error_message}", file=sys.stderr)
    if payload and isinstance(payload, dict):
        return payload, elapsed, margin, error_message
    return _zero_result(pair_id), elapsed, 0.0, error_message


def _read_input_rows(input_csv):
    with open(input_csv, newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header row: {input_csv}")

        rows = []
        for raw_row in reader:
            pair_id = _canonical_value(raw_row, "pair_id", "pairId")
            if pair_id is None:
                raise ValueError(f"Missing pair_id column in input CSV: {input_csv}")

            reference_value = _canonical_value(
                raw_row,
                "reference_path", "ref_path", "reference", "reference_image", "reference_png"
            )
            search_value = _canonical_value(
                raw_row,
                "search_path", "search", "search_image", "search_png"
            )
            if reference_value is None or search_value is None:
                raise ValueError(
                    "Input CSV must include reference_path/ref_path and search_path/search columns. "
                    f"Found columns: {reader.fieldnames}"
                )

            rows.append({
                "pair_id": str(pair_id).strip(),
                "reference_path": _resolve_csv_path(reference_value, Path(input_csv).resolve().parent),
                "search_path": _resolve_csv_path(search_value, Path(input_csv).resolve().parent),
            })
    return rows


def _write_predictions(output_csv, rows_in_order):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows_in_order:
            record = {
                "pair_id": row["pair_id"],
                "x": int(row["x"]),
                "y": int(row["y"]),
                "theta": float(row["theta"]),
                "scale": float(row["scale"]),
                "found": int(row["found"]),
                "score": float(row["score"]),
            }
            writer.writerow(record)


def _write_time_log(output_csv, rows_in_order):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=TIME_LOG_FIELDS)
        writer.writeheader()
        for row in rows_in_order:
            writer.writerow({
                "pair_id": row["pair_id"],
                "time_sec": float(row["time_sec"]),
                "best_score": float(row["best_score"]),
                "top2_margin": float(row["top2_margin"]),
                "found": int(row["found"]),
                "status": row["status"],
                "error": row.get("error", ""),
            })


def main():
    parser = argparse.ArgumentParser(description="Batch localization register")
    parser.add_argument("--input", required=True, help="Input CSV with pair_id and reference/search paths")
    parser.add_argument("--output", required=True, help="Output CSV path for predictions")
    args = parser.parse_args()

    start = time.perf_counter()

    rows = _read_input_rows(args.input)
    ordered_pair_ids = []
    seen = set()
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in seen:
            continue
        seen.add(pair_id)
        ordered_pair_ids.append(pair_id)

    results_by_pair = {}
    debug_rows = []
    total_pairs = len(ordered_pair_ids)
    for idx, pair_id in enumerate(ordered_pair_ids, start=1):
        print(f"pair {idx}/{total_pairs}", flush=True)
        row = next(r for r in rows if r["pair_id"] == pair_id)
        prediction, elapsed_sec, top2_margin, error = _process_pair(pair_id, row["reference_path"], row["search_path"])
        results_by_pair[pair_id] = prediction
        debug_rows.append({
            "pair_id": pair_id,
            "time_sec": elapsed_sec,
            "best_score": float(prediction["score"]),
            "top2_margin": top2_margin,
            "found": int(prediction["found"]),
            "status": "error" if error else "ok",
            "error": error or "",
        })

    ordered_rows = [results_by_pair[pair_id] for pair_id in ordered_pair_ids]
    _write_predictions(args.output, ordered_rows)
    timings_csv = str(Path(args.output).with_name(Path(args.output).stem + "_timings.csv"))
    _write_time_log(timings_csv, debug_rows)

    zero_margin_pairs = [r for r in debug_rows if r["status"] == "ok" and int(results_by_pair[r["pair_id"]]["found"]) == 0]
    if zero_margin_pairs:
        print("Zero-found top2_margin values:", file=sys.stderr)
        for item in zero_margin_pairs:
            print(
                f"pair_id={item['pair_id']} best_score={item['best_score']:.6f} "
                f"top2_margin={item['top2_margin']} time_sec={item['time_sec']:.3f}",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - start
    print(f"rows_written={len(ordered_rows)}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"output_csv={args.output}")
    print(f"timings_csv={timings_csv}")


if __name__ == "__main__":
    main()
