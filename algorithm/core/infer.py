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


def multi_peak_search(image, template,
                       angle_range=3, angle_step=1,
                       scale_center=0.10, scale_range=0.02, scale_step=0.005,
                       tie_tolerance=0.005, peaks_per_scale=6,
                       cluster_dist=20):
    img_h, img_w = image.shape[:2]
    img_center = np.array([img_w / 2, img_h / 2])
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    scales = np.arange(scale_center - scale_range, scale_center + scale_range + scale_step, scale_step)
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
    cluster_best = {}
    cluster_coords = {}  # track (x, y) for each cluster key for returning candidates
    for (x, y, score, angle, scale) in all_candidates:
        key = (round(x / cluster_dist), round(y / cluster_dist))
        if key not in cluster_best or score > cluster_best[key]:
            cluster_best[key] = score
            cluster_coords[key] = (x, y)
    sorted_clusters = sorted(cluster_best.items(), key=lambda kv: kv[1], reverse=True)
    sorted_scores = [score for _, score in sorted_clusters]
    if len(sorted_scores) >= 2:
        ratio = sorted_scores[1] / sorted_scores[0]
    else:
        ratio = 0.0  # only one distinct location found at all -- unambiguous

    # Collect top 2-3 candidates for low-confidence reporting
    top_candidates = []
    for i, (key, score) in enumerate(sorted_clusters[:3]):  # top 3 clusters
        x, y = cluster_coords[key]
        top_candidates.append((int(round(x)), int(round(y)), round(score, 3)))

    return best, ratio, top_candidates


def run_inference(reference_path, search_path):
    reference = cv2.imread(reference_path)
    search = cv2.imread(search_path)
    if reference is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_path}")
    if search is None:
        raise FileNotFoundError(f"Could not read search image: {search_path}")
    (x, y, score, angle, scale), ratio, candidates = multi_peak_search(search, reference)
    confidence = "LOW" if ratio > RATIO_THRESHOLD else "HIGH"
    return int(round(x)), int(round(y)), score, ratio, confidence, angle, scale, candidates


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