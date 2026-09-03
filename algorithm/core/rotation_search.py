import cv2
import numpy as np
from ncc import ncc_match

def rotate_image(img, angle):
    """Rotates an image around its center, keeping the same canvas size."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def rotation_search(image, template, angle_range=30, angle_step=5):
    """
    Tries rotating the TEMPLATE across a range of angles,
    runs NCC at each, and keeps the best-scoring result.

    angle_range: search from -angle_range to +angle_range degrees
    angle_step: coarse step size in degrees
    """
    best_score = -1
    best_result = None

    for angle in np.arange(-angle_range, angle_range + angle_step, angle_step):
        rotated_template = rotate_image(template, angle)
        x, y, score = ncc_match(image, rotated_template)

        if score > best_score:
            best_score = score
            best_result = (x, y, score, angle)

    return best_result  # (x, y, score, best_angle)


if __name__ == "__main__":
    import json

    with open("../tests/synthetic_data/pair_1_meta.json") as f:
        meta = json.load(f)

    image = cv2.imread("../tests/" + meta["image_path"])
    template = cv2.imread("../tests/" + meta["template_path"])

    pred_x, pred_y, score, best_angle = rotation_search(image, template)

    gt_x, gt_y = meta["gt_x"], meta["gt_y"]
    error = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5

    print(f"Ground truth center: ({gt_x}, {gt_y}), true angle: {meta['gt_angle']:.1f}")
    print(f"Predicted center:    ({pred_x}, {pred_y}), best found angle: {best_angle:.1f}")
    print(f"NCC score: {score:.3f}")
    print(f"Pixel error: {error:.2f}")