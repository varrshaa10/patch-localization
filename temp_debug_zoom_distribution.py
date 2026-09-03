import csv
import sys
from pathlib import Path

import cv2
import numpy as np

repo = Path(__file__).resolve().parent
sys.path.insert(0, str(repo / 'algorithm' / 'core'))

from infer import coarse_to_fine_search
from ncc import ncc_match_multi
from rotation_search import rotate_image
from scale_rotation_search import scale_template


def main():
    gt_csv = repo / 'algorithm' / 'tests' / 'synthetic_data_phase2' / 'ground_truth_combined.csv'
    if not gt_csv.exists():
        raise FileNotFoundError(f'Missing ground truth file: {gt_csv}')

    with gt_csv.open(newline='') as f:
        rows = list(csv.DictReader(f))

    output_csv = repo / 'score_calibration.csv'
    with output_csv.open('w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['pair_id', 'category', 'raw_ncc_score', 'best_minus_median'])

        for row in rows:
            pair_id = row['pair_id']
            category = row.get('category', '')

            pair_dir = repo / 'algorithm' / 'tests' / 'synthetic_data_phase2' / f'pair_{pair_id}'
            ref_path = pair_dir / 'reference.png'
            search_path = pair_dir / 'search.png'
            if not ref_path.exists() or not search_path.exists():
                writer.writerow([pair_id, category, 'MISSING', 'MISSING'])
                continue

            image = cv2.imread(str(search_path))
            template = cv2.imread(str(ref_path))
            if image is None or template is None:
                writer.writerow([pair_id, category, 'READ_FAIL', 'READ_FAIL'])
                continue

            best, _, _, _, _ = coarse_to_fine_search(
                image,
                template,
                angle_range=5,
                angle_step=1,
                zoom_min=8.0,
                zoom_max=12.0,
                zoom_step=0.5,
            )
            raw_ncc_score = float(best[2])

            angle = float(best[3])
            scale = float(best[4])
            rotated = rotate_image(template, angle)
            scaled_template = scale_template(rotated, scale)
            peaks = ncc_match_multi(image, scaled_template, num_peaks=20, min_distance=15)
            peak_scores = [float(score) for _, _, score in peaks]
            if peak_scores:
                best_idx = int(np.argmax(peak_scores))
                best_peak_score = float(peak_scores[best_idx])
                remaining = peak_scores[:best_idx] + peak_scores[best_idx + 1:]
                median_of_rest = float(np.median(remaining)) if remaining else 0.0
                best_minus_median = best_peak_score - median_of_rest
            else:
                best_minus_median = 0.0

            writer.writerow([
                pair_id,
                category,
                f'{raw_ncc_score:.6f}',
                f'{best_minus_median:.6f}',
            ])

    print(f'Wrote calibration scores to {output_csv}')


if __name__ == '__main__':
    main()
