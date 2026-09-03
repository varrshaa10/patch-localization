import cv2
import numpy as np


def ncc_match(image, template):
    """
    Runs normalized cross-correlation and returns the best match location.
    Input: grayscale image and template (numpy arrays, uint8 or float32)
    Output: (x, y) = predicted CENTER of the match, and the score
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    top_left_x, top_left_y = max_loc
    h, w = template.shape[:2]
    center_x = top_left_x + w // 2
    center_y = top_left_y + h // 2
    return center_x, center_y, max_val


if __name__ == "__main__":
    import json
    with open("../tests/synthetic_data/pair_0_meta.json") as f:
        meta = json.load(f)
    image = cv2.imread("../tests/" + meta["image_path"])
    template = cv2.imread("../tests/" + meta["template_path"])
    pred_x, pred_y, score = ncc_match(image, template)
    gt_x, gt_y = meta["gt_x"], meta["gt_y"]
    error = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5
    print(f"Ground truth center: ({gt_x}, {gt_y})")
    print(f"Predicted center:    ({pred_x}, {pred_y})")
    print(f"NCC score: {score:.3f}")
    print(f"Pixel error: {error:.2f}")


def ncc_match_multi(image, template, num_peaks=8, min_distance=15):
    """
    Like ncc_match, but returns the top `num_peaks` local maxima from the
    correlation surface (not just the single global best), suppressing
    peaks that are within `min_distance` px of an already-kept peak.
    Needed for periodic layouts where many near-identical matches exist.
    Returns a list of (center_x, center_y, score).
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]
    result_copy = result.copy()
    peaks = []
    for _ in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)
        top_left_x, top_left_y = max_loc
        center_x = top_left_x + w // 2
        center_y = top_left_y + h // 2
        peaks.append((center_x, center_y, max_val))
        y0 = max(0, top_left_y - min_distance)
        y1 = min(result_copy.shape[0], top_left_y + min_distance)
        x0 = max(0, top_left_x - min_distance)
        x1 = min(result_copy.shape[1], top_left_x + min_distance)
        result_copy[y0:y1, x0:x1] = -1
    return peaks