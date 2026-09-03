import cv2
import numpy as np
from ncc import ncc_match
from rotation_search import rotate_image

def scale_template(template, scale):
    h, w = template.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(template, (new_w, new_h))

def scale_rotation_search(image, template,
                           angle_range=3, angle_step=1,
                           zoom_min=8.0, zoom_max=12.0, zoom_step=0.5):
    """
    Search a direct zoom range [zoom_min, zoom_max] and convert to the
    internal NCC scale values as 1.0 / zoom.
    """
    best_score = -1
    best_result = None

    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    zooms = np.arange(zoom_min, zoom_max + zoom_step, zoom_step)
    scales = [1.0 / zoom for zoom in zooms]

    for angle in angles:
        rotated = rotate_image(template, angle)
        for scale in scales:
            scaled = scale_template(rotated, scale)
            if scaled.shape[0] < 4 or scaled.shape[1] < 4:
                continue
            if scaled.shape[0] >= image.shape[0] or scaled.shape[1] >= image.shape[1]:
                continue
            x, y, score = ncc_match(image, scaled)
            if score > best_score:
                best_score = score
                best_result = (x, y, score, angle, scale)

    return best_result
if __name__ == "__main__":
    import json

    with open("../tests/synthetic_data/pair_3_meta.json") as f:
        meta = json.load(f)

    image = cv2.imread("../tests/" + meta["image_path"])
    template = cv2.imread("../tests/" + meta["template_path"])

    pred_x, pred_y, score, best_angle, best_scale = scale_rotation_search(image, template)

    gt_x, gt_y = meta["gt_x"], meta["gt_y"]
    error = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5

    print(f"Ground truth: ({gt_x}, {gt_y})")
    print(f"Predicted:    ({pred_x}, {pred_y}), angle={best_angle:.1f}, scale={best_scale:.3f}")
    print(f"NCC score: {score:.3f}")
    print(f"Pixel error: {error:.2f}")