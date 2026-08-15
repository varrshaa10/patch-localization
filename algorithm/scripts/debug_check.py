import cv2
from ncc import ncc_match_multi
from rotation_search import rotate_image
from scale_rotation_search import scale_template

image = cv2.imread("../tests/synthetic_data/pair_0_search.png")
template = cv2.imread("../tests/synthetic_data/pair_0_reference.png")

# use the winning angle/scale from the last run
rotated = rotate_image(template, 2.0)
scaled = scale_template(rotated, 0.10)

peaks = ncc_match_multi(image, scaled, num_peaks=15, min_distance=15)
peaks.sort(key=lambda p: -p[2])

print("Top peaks (x, y, score):")
for p in peaks:
    print(p)