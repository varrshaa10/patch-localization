import json
import cv2
from ncc import ncc_match

for i in range(5):
    with open(f"../tests/synthetic_data/pair_{i}_meta.json") as f:
        meta = json.load(f)

    image = cv2.imread("../tests/" + meta["image_path"])
    template = cv2.imread("../tests/" + meta["template_path"])

    pred_x, pred_y, score = ncc_match(image, template)
    gt_x, gt_y = meta["gt_x"], meta["gt_y"]
    error = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5

    print(f"pair {i}: angle={meta['gt_angle']:.1f} scale={meta['gt_scale']:.2f} "
          f"-> error={error:.2f}px, score={score:.3f}")