import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OFFICIAL_DIR = ROOT / 'algorithm' / 'tests' / 'official_phase2'
PRED_PATH = OFFICIAL_DIR / 'predictions.csv'
GT_PATH = OFFICIAL_DIR / 'ground_truth.csv'
TIMINGS_PATH = OFFICIAL_DIR / 'predictions_timings.csv'

SET_GROUPS = {
    'Set A': [f'p{i:03d}' for i in range(1, 9)],
    'Set B': [f'p{i:03d}' for i in range(9, 15)],
    'Set C': [f'p{i:03d}' for i in range(15, 19)],
    'Set D': [f'p{i:03d}' for i in range(19, 21)],
}

REJECTION_THRESHOLD = 0.55


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def credit_from_error(err):
    if err is None:
        return 0.0
    if err <= 1.0:
        return 1.00
    if err <= 2.0:
        return 0.80
    if err <= 3.0:
        return 0.60
    if err <= 5.0:
        return 0.40
    return 0.00


def detection_from_prediction(pred_row):
    if pred_row is None:
        return False
    if 'found' in pred_row:
        found_value = pred_row.get('found', '')
        if found_value not in (None, ''):
            return int(float(found_value)) == 1
    score_value = pred_row.get('score', '0')
    return float(score_value) >= REJECTION_THRESHOLD


def evaluate():
    preds = {row['pair_id']: row for row in read_csv(PRED_PATH)}
    gt_rows = {row['pair_id']: row for row in read_csv(GT_PATH)}
    timing_rows = read_csv(TIMINGS_PATH)

    present_credits = []
    present_errors = []
    set_stats = {name: {'credits': [], 'errors': []} for name in SET_GROUPS}

    tp = fp = fn = tn = 0
    scale_errors = []
    theta_errors = []

    for pair_id in sorted(gt_rows):
        gt_row = gt_rows[pair_id]
        pred_row = preds.get(pair_id)
        if pred_row is None:
            raise KeyError(f'Missing prediction row for {pair_id}')

        present = int(float(gt_row['present']))
        found = detection_from_prediction(pred_row)
        pred_x = float(pred_row.get('x', 0.0) or 0.0)
        pred_y = float(pred_row.get('y', 0.0) or 0.0)
        gt_x = float(gt_row['x'])
        gt_y = float(gt_row['y'])
        pred_theta = float(pred_row.get('theta', 0.0) or 0.0)
        gt_theta = float(gt_row['theta'])
        pred_scale = float(pred_row.get('scale', 0.0) or 0.0)
        gt_scale = float(gt_row['scale'])

        if present == 1:
            if found:
                err = math.hypot(pred_x - gt_x, pred_y - gt_y)
                credit = credit_from_error(err)
                present_errors.append(err)
                present_credits.append(credit)
                tp += 1
                if credit > 0.0:
                    scale_errors.append(abs(pred_scale - gt_scale) / gt_scale)
                    theta_errors.append(abs(pred_theta - gt_theta))
            else:
                credit = 0.0
                fn += 1

            for set_name, ids in SET_GROUPS.items():
                if pair_id in ids:
                    set_stats[set_name]['credits'].append(credit)
                    if found:
                        set_stats[set_name]['errors'].append(math.hypot(pred_x - gt_x, pred_y - gt_y))
                    break
        else:
            if found:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    timings = [float(row['time_sec']) for row in timing_rows]
    median_runtime = statistics.median(timings)
    max_runtime = max(timings)

    print('Official Phase 2 scoring report')
    print('==============================')
    print(f'Predictions file: {PRED_PATH}')
    print(f'Ground truth file: {GT_PATH}')
    print(f'Timings file: {TIMINGS_PATH}')
    print()

    for set_name, ids in SET_GROUPS.items():
        credits = set_stats[set_name]['credits']
        errors = set_stats[set_name]['errors']
        mean_credit = statistics.mean(credits) if credits else 0.0
        median_err = statistics.median(errors) if errors else 'N/A'
        if median_err == 'N/A':
            median_err_text = 'N/A'
        else:
            median_err_text = f'{median_err:.3f}px'
        print(f'{set_name}:')
        print(f'  pairs: {len(ids)}')
        print(f'  mean credit: {mean_credit:.3f}')
        print(f'  median pixel error: {median_err_text}')
        print()

    overall_mean_credit = statistics.mean(present_credits) if present_credits else 0.0
    print('Overall present-pair statistics:')
    print(f'  mean credit: {overall_mean_credit:.3f}')
    print(f'  median pixel error: {statistics.median(present_errors) if present_errors else "N/A"}')
    print()

    print('Rejection metrics (score threshold = 0.55):')
    print(f'  precision: {precision:.6f}')
    print(f'  recall:    {recall:.6f}')
    print(f'  F1:        {f1:.6f}')
    print(f'  TP={tp} FP={fp} FN={fn} TN={tn}')
    print()

    print('Pose accuracy on positive-credit detections:')
    if scale_errors:
        print(f'  scale % error median: {statistics.median(scale_errors) * 100:.3f}%')
        print(f'  scale % error worst:  {max(scale_errors) * 100:.3f}%')
        print(f'  theta abs error median: {statistics.median(theta_errors):.3f}°')
        print(f'  theta abs error worst:  {max(theta_errors):.3f}°')
    else:
        print('  scale % error median: N/A')
        print('  scale % error worst:  N/A')
        print('  theta abs error median: N/A')
        print('  theta abs error worst:  N/A')
    print()

    print('Runtime (timings CSV):')
    print(f'  median time_sec: {median_runtime:.6f}s')
    print(f'  max time_sec:    {max_runtime:.6f}s')
    print()

    print('Organizer reference baseline:')
    print('  Set A credit: 1.000')
    print('  Set B credit: 0.467')
    print('  Set D credit: 1.000')
    print('  Rejection F1: 0.897')
    print('  Pose scale worst/median: 3.0% / 1.0%')
    print('  Pose theta worst/median: 1.10° / 0.35°')


if __name__ == '__main__':
    evaluate()
