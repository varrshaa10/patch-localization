"""
Drift-Sense standalone localization inference script.
Usage:
    python infer.py --reference path/to/ref.png --search path/to/search.png
Outputs the predicted center (x, y) in pixel coordinates within the search image.
"""
import argparse
import cv2
import numpy as np
from ncc import ncc_match, ncc_match_multi
from rotation_search import rotate_image
from scale_rotation_search import scale_template

# Ambiguity ratio threshold: second-best distinct location's score divided
# by the best location's score. Close to 1.0 = near-tied = ambiguous.
# Calibrate against your own pairs -- see note below.
RATIO_THRESHOLD = 0.995
ANGLE_RANGE = 3
ANGLE_STEP = 1
ZOOM_MIN = 8.0
ZOOM_MAX = 12.0
ZOOM_STEP = 0.5


def _grid_values(start, stop, step):
    return np.arange(start, stop + step, step)


def _prepare_template_variant(template, angle, scale):
    rotated = rotate_image(template, angle)
    return scale_template(rotated, scale)


def _candidate_clusters(all_candidates, cluster_dist=20):
    cluster_best = {}
    cluster_coords = {}
    for (x, y, score, angle, scale) in all_candidates:
        key = (round(x / cluster_dist), round(y / cluster_dist))
        if key not in cluster_best or score > cluster_best[key]:
            cluster_best[key] = score
            cluster_coords[key] = (x, y)
    sorted_clusters = sorted(cluster_best.items(), key=lambda kv: kv[1], reverse=True)
    sorted_scores = [score for _, score in sorted_clusters]
    top2_margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else sorted_scores[0]
    top_candidates = []
    for key, score in sorted_clusters[:3]:
        x, y = cluster_coords[key]
        top_candidates.append((int(round(x)), int(round(y)), round(score, 3)))
    return sorted_clusters, top_candidates, top2_margin


def coarse_to_fine_search(image, template,
                         angle_range=3, angle_step=1,
                         zoom_min=8.0, zoom_max=12.0, zoom_step=0.5,
                         tie_tolerance=0.005, peaks_per_scale=6,
                         cluster_dist=20):
    img_h, img_w = image.shape[:2]
    img_center = np.array([img_w / 2, img_h / 2])
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    zooms = np.arange(zoom_min, zoom_max + zoom_step, zoom_step)
    scales = [1.0 / zoom for zoom in zooms]
    all_candidates = []
    best_score = -1
    for angle in angles:
        rotated = rotate_image(template, angle)
        for scale in scales:
            scaled = scale_template(rotated, scale)
            if scaled.shape[0] < 4 or scaled.shape[1] < 4:
                continue
            if scaled.shape[0] >= img_h or scaled.shape[1] >= img_w:
                continue
            peaks = ncc_match_multi(image, scaled, num_peaks=peaks_per_scale, min_distance=15)
            for (x, y, score) in peaks:
                all_candidates.append((x, y, score, angle, scale))
                if score > best_score:
                    best_score = score
    if not all_candidates:
        raise RuntimeError("No valid candidates found -- check image/template sizes.")

    near_best = [c for c in all_candidates if c[2] >= best_score - tie_tolerance]
    best = min(near_best, key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - img_center))
    sorted_clusters, top_candidates, top2_margin = _candidate_clusters(all_candidates, cluster_dist=cluster_dist)
    ratio = 0.0 if len(sorted_clusters) < 2 else sorted_clusters[1][1] / sorted_clusters[0][1]
    return best, ratio, top_candidates, sorted_clusters, top2_margin


def full_grid_search(image, template,
                    angle_range=3, angle_step=1,
                    zoom_min=8.0, zoom_max=12.0, zoom_step=0.5,
                    tie_tolerance=0.005, peaks_per_scale=6,
                    cluster_dist=20):
    return coarse_to_fine_search(image, template, angle_range=angle_range, angle_step=angle_step,
                                zoom_min=zoom_min, zoom_max=zoom_max, zoom_step=zoom_step,
                                tie_tolerance=tie_tolerance, peaks_per_scale=peaks_per_scale,
                                cluster_dist=cluster_dist)


def multi_peak_search(image, template,
                       angle_range=3, angle_step=1,
                       zoom_min=8.0, zoom_max=12.0, zoom_step=0.5,
                       tie_tolerance=0.005, peaks_per_scale=6,
                       cluster_dist=20):
    img_h, img_w = image.shape[:2]
    img_center = np.array([img_w / 2, img_h / 2])
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    zooms = np.arange(zoom_min, zoom_max + zoom_step, zoom_step)
    scales = [1.0 / zoom for zoom in zooms]
    all_candidates = []  # (x, y, score, angle, scale)
    best_score = -1
    for angle in angles:
        rotated = rotate_image(template, angle)
        for scale in scales:
            scaled = scale_template(rotated, scale)
            if scaled.shape[0] < 4 or scaled.shape[1] < 4:
                continue
            if scaled.shape[0] >= img_h or scaled.shape[1] >= img_w:
                continue
            peaks = ncc_match_multi(image, scaled, num_peaks=peaks_per_scale, min_distance=15)
            for (x, y, score) in peaks:
                all_candidates.append((x, y, score, angle, scale))
                if score > best_score:
                    best_score = score
    if not all_candidates:
        raise RuntimeError("No valid candidates found -- check image/template sizes.")

    near_best = [c for c in all_candidates if c[2] >= best_score - tie_tolerance]
    def dist_to_center(c):
        return np.linalg.norm(np.array([c[0], c[1]]) - img_center)
    best = min(near_best, key=dist_to_center)

    # --- Ambiguity check: collapse all candidates into distinct spatial
    # clusters (grid-snap by cluster_dist), keep each cluster's best score,
    # then compare the top two distinct clusters. ---
    sorted_clusters, top_candidates, top2_margin = _candidate_clusters(all_candidates, cluster_dist=cluster_dist)
    sorted_scores = [score for _, score in sorted_clusters]
    if len(sorted_scores) >= 2:
        ratio = sorted_scores[1] / sorted_scores[0]
    else:
        ratio = 0.0

    return best, ratio, top_candidates


def _split_template_into_grid(template, grid_size=3):
    if template is None:
        return []
    gray = template
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return []

    patch_h = max(1, h // grid_size)
    patch_w = max(1, w // grid_size)
    parts = []
    for row in range(grid_size):
        y0 = min(row * patch_h, h)
        y1 = min((row + 1) * patch_h, h)
        for col in range(grid_size):
            x0 = min(col * patch_w, w)
            x1 = min((col + 1) * patch_w, w)
            if y1 <= y0 or x1 <= x0:
                continue
            subpatch = gray[y0:y1, x0:x1]
            cx = (x0 + x1) / 2.0 - (w / 2.0)
            cy = (y0 + y1) / 2.0 - (h / 2.0)
            parts.append(((cx, cy), subpatch))
    return parts


def _subregion_consistency_score(candidate, search, template, local_window=5, grid_size=3):
    if candidate is None or len(candidate) < 5:
        return 0.0
    x, y, _, _, _ = candidate
    if search is None or template is None:
        return 0.0
    gray = search
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    template_gray = template
    if len(template_gray.shape) == 3:
        template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)

    patches = _split_template_into_grid(template_gray, grid_size=grid_size)
    if not patches:
        return 0.0

    offsets = []
    for (dx, dy), patch in patches:
        expected_x = x + dx
        expected_y = y + dy
        roi_x0 = max(0, int(round(expected_x - local_window)))
        roi_x1 = min(gray.shape[1], int(round(expected_x + local_window + 1)))
        roi_y0 = max(0, int(round(expected_y - local_window)))
        roi_y1 = min(gray.shape[0], int(round(expected_y + local_window + 1)))
        if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
            return 0.0
        roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
        if roi.shape[0] < 1 or roi.shape[1] < 1:
            return 0.0
        result = cv2.matchTemplate(roi, patch, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        wx = max_loc[0] + patch.shape[1] / 2.0
        wy = max_loc[1] + patch.shape[0] / 2.0
        offset_x = (roi_x0 + wx) - expected_x
        offset_y = (roi_y0 + wy) - expected_y
        offsets.append((offset_x, offset_y))

    if not offsets:
        return 0.0

    centroid_x = float(np.mean([o[0] for o in offsets]))
    centroid_y = float(np.mean([o[1] for o in offsets]))
    variance = float(np.mean([
        (o[0] - centroid_x) ** 2 + (o[1] - centroid_y) ** 2
        for o in offsets
    ]))
    return 1.0 / (1.0 + variance)


def rerank_peak_candidates_weighted_ncc(candidates, search, reference):
    if not candidates:
        return []
    scored = []
    for candidate in candidates:
        x, y, raw_score, angle, scale = candidate
        consistency = _subregion_consistency_score(candidate, search, reference)
        weighted_score = float(raw_score) * (0.75 + 0.25 * consistency)
        scored.append(((x, y, weighted_score, angle, scale), weighted_score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [candidate for candidate, _ in scored]


def rerank_peak_candidates_phase_correlation(candidates, search, reference):
    if not candidates:
        return []
    scored = []
    for candidate in candidates:
        x, y, raw_score, angle, scale = candidate
        phase_like = float(raw_score) + 0.1 * _subregion_consistency_score(candidate, search, reference)
        scored.append(((x, y, phase_like, angle, scale), phase_like))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [candidate for candidate, _ in scored]


def rerank_peak_candidates_subregion_consistency(candidates, search, reference, local_window=5, grid_size=3):
    if not candidates:
        return []
    scored = []
    for candidate in candidates:
        x, y, raw_score, angle, scale = candidate
        consistency = _subregion_consistency_score(candidate, search, reference, local_window=local_window, grid_size=grid_size)
        scored.append(((x, y, raw_score, angle, scale), consistency))
    scored.sort(key=lambda item: (item[1], item[0][2]), reverse=True)
    return [candidate for candidate, _ in scored]


def _vote_cluster_locations(locations, tolerance=10.0):
    if not locations:
        return []
    clusters = []
    for loc in locations:
        placed = False
        for cluster in clusters:
            if np.hypot(loc[0] - cluster["center"][0], loc[1] - cluster["center"][1]) <= tolerance:
                cluster["points"].append(loc)
                cluster["center"] = (
                    float(np.mean([p[0] for p in cluster["points"]])),
                    float(np.mean([p[1] for p in cluster["points"]])),
                )
                placed = True
                break
        if not placed:
            clusters.append({"center": loc, "points": [loc]})
    return sorted(clusters, key=lambda c: len(c["points"]), reverse=True)


def majority_vote_ensemble(search, reference, candidate_pool=None, tolerance=10.0):
    """Two-method majority vote: weighted-NCC and phase-correlation agreement,
    else fall back to the raw NCC baseline.
    """
    if candidate_pool is None:
        candidate_pool = []
    raw_best, _, _, _, _ = coarse_to_fine_search(search, reference)

    agreement_points = []
    if candidate_pool:
        weighted_top = rerank_peak_candidates_weighted_ncc(candidate_pool, search, reference)
        if weighted_top:
            agreement_points.append((float(weighted_top[0][0]), float(weighted_top[0][1])))

        phase_top = rerank_peak_candidates_phase_correlation(candidate_pool, search, reference)
        if phase_top:
            agreement_points.append((float(phase_top[0][0]), float(phase_top[0][1])))

    if len(agreement_points) >= 2:
        clusters = _vote_cluster_locations(agreement_points, tolerance=tolerance)
        if clusters and len(clusters[0]["points"]) >= 2:
            x = float(np.mean([p[0] for p in clusters[0]["points"]]))
            y = float(np.mean([p[1] for p in clusters[0]["points"]]))
            return (x, y, float(raw_best[2]), float(raw_best[3]), float(raw_best[4]))

    return raw_best


def ensemble_rerank_candidates(candidates, search, reference, tolerance=10.0):
    if not candidates:
        return []
    top = majority_vote_ensemble(search, reference, candidate_pool=candidates, tolerance=tolerance)
    return [top]


def run_inference(reference_path, search_path):
    reference = cv2.imread(reference_path)
    search = cv2.imread(search_path)
    if reference is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_path}")
    if search is None:
        raise FileNotFoundError(f"Could not read search image: {search_path}")
    (x, y, score, angle, scale), ratio, candidates = multi_peak_search(search, reference)
    top2_margin = max(0.0, score - (ratio * score if ratio > 0 else 0.0))
    near_best_count = len(candidates)
    confidence = "LOW" if ratio > RATIO_THRESHOLD else "HIGH"
    return int(round(x)), int(round(y)), score, ratio, confidence, angle, scale, candidates, near_best_count, top2_margin


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense localization inference")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    args = parser.parse_args()
    x, y, score, ratio, confidence, angle, scale, candidates = run_inference(args.reference, args.search)
    print(f"Predicted center (x, y): ({x}, {y})")
    print(f"NCC score: {score:.3f}")
    print(f"Ambiguity ratio: {ratio:.3f}  Confidence: {confidence}")
    if confidence == "LOW":
        print("  -> possible periodic ambiguity: multiple near-tied match locations detected")
        print(f"  Candidate locations (top {len(candidates)}):")
        for i, (cx, cy, cscore) in enumerate(candidates, 1):
            print(f"    {i}. ({cx}, {cy}) score {cscore}")
    print(f"Best angle: {angle:.1f} deg, best scale: {scale:.3f}")


if __name__ == "__main__":
    main()