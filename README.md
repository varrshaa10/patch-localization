# Patch Localization: Classical NCC-Based SEM Site Relocalization

## Project Overview

**Patch-Localization** is a classical computer vision project (no machine learning, no neural networks, no training) that solves the **drift-sense site relocalization problem**: given a reference image crop (patch) of a semiconductor microstructure and a larger, noisier search image, find where the reference is located within the search image.

The solution uses **normalized cross-correlation (NCC)** template matching combined with multi-scale and multi-rotation search, augmented with an **ambiguity-ratio confidence metric** inspired by SIFT's ratio test. This allows detection of inherently ambiguous cases (periodic patterns without disambiguating features) without training-based approaches.

**Key characteristics:**
- Classical computer vision approach (no DL models, training, or model weights)
- Synthetic dataset: 40 image pairs (20 DRAM + 20 FinFET microstructures)
- 80% accuracy on periodic-ambiguity test suite
- Confidence detection: 100% agreement with ground truth correctness
- Includes 8 intentional hard cases (periodic patterns, no navigation marker)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/varrshaa10/patch-localization
cd patch-localization
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv

# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` — Numerical computation
- `opencv-python` — Image processing (template matching, transformations)
- `pillow` — Image I/O
- `matplotlib` — Plotting (optional, for visualization)

---

## Usage

### Generate Synthetic Dataset

```bash
cd algorithm
python dataset/generate_dataset.py [--architecture {dram|finfet|both}] [--num_pairs N] [--output_dir DIR] [--color] [--jitter]
```

**Arguments:**
- `--architecture`: Microstructure style (default: `both`) — `dram`, `finfet`, or `both`
- `--num_pairs`: Number of image pairs to generate (default: `30`)
- `--output_dir`: Output directory for images and ground-truth CSV (default: `../tests/synthetic_data`)
- `--color`: Generate RGB color variants instead of grayscale (useful for testing color-tinted SEM patterns)
- `--jitter`: Apply ±10–20% random size jitter to DRAM/FinFET features, simulating process variation

**Examples:**
```bash
python dataset/generate_dataset.py --architecture both --num_pairs 40 --output_dir ./tests/synthetic_data
python dataset/generate_dataset.py --architecture both --num_pairs 5 --output_dir ./tests/synthetic_data_rgb --color
python dataset/generate_dataset.py --architecture both --num_pairs 40 --output_dir ./tests/synthetic_data --jitter
```

**Output:**
- Generated images saved as PNG files in the specified output directory
- Ground truth locations stored in `<output_dir>/ground_truth.csv` with columns:
  - `pair_id`, `architecture`, `ref_path`, `search_path`, `gt_x`, `gt_y`, `angle`, `scale`, `add_marker`, `mode`

---

### Run Inference on a Single Pair

```bash
cd algorithm/core
python infer.py --reference <path/to/reference.png> --search <path/to/search.png>
```

**Example:**
```bash
python infer.py --reference ../tests/synthetic_data/pair_0_reference.png --search ../tests/synthetic_data/pair_0_search.png
```

**Sample Output (LOW confidence / ambiguous case):**
```
Predicted center (x, y): (601, 245)
NCC score: 0.963
Ambiguity ratio: 1.000  Confidence: LOW
  -> possible periodic ambiguity: multiple near-tied match locations detected
  Candidate locations (top 3):
    1. (631, 245) score 0.963
    2. (601, 245) score 0.963
    3. (631, 215) score 0.963
Best angle: 0.0 deg, best scale: 0.100
```

**Sample Output (HIGH confidence / clear case):**
```
Predicted center (x, y): (274, 62)
NCC score: 0.853
Ambiguity ratio: 0.913  Confidence: HIGH
Best angle: 3.0 deg, best scale: 0.100
```

**Output fields:**
- `Predicted center (x, y)`: Pixel coordinates of the match within the search image
- `NCC score`: Normalized correlation coefficient (0–1, higher is better)
- `Ambiguity ratio`: Ratio of second-best cluster's score to best cluster's score
  - Ratio ≈ 1.000 → LOW confidence (periodic ambiguity, multiple indistinguishable locations)
  - Ratio << 1.000 → HIGH confidence (clear, unambiguous match)
- `Confidence`: Label assigned by comparing `ambiguity_ratio` to threshold (0.995)
- `Candidate locations`: Top 2–3 near-tied spatial hypotheses shown only for LOW-confidence matches
- `Best angle`: Best-fit rotation angle (−3° to +3°)
- `Best scale`: Best-fit scale factor (~0.08–0.12)

---

### Run Batch Evaluation

```bash
cd algorithm/core
python batch_eval.py
```

This evaluates the matcher on all 40 ground-truth pairs, computes per-pair pixel error, and generates a summary report.

**Output:**
```
Accuracy within 1px: 80.0%
Accuracy within 2px: 80.0%
...
Accuracy within 50px: 80.0%

Ambiguity ratio distribution:
Ratio on FAILED pairs (n=8): min=1.000, max=1.000, mean=1.000
Ratio on PASSED pairs (n=32): min=0.980, max=0.990, mean=0.985
```

Results are written to `batch_results.csv`.

---
### Reproduced Evaluation

Evaluation on 40 generated grayscale reference/search pairs:

| Metric | Result |
|---|---:|
| Accuracy @ 1 px | 80.0% |
| Accuracy @ 2 px | 80.0% |
| Accuracy @ 3 px | 80.0% |
| Accuracy @ 5 px | 80.0% |
| Median localization error | 0.45 px |
| Mean localization error | 52.72 px |
| Mean inference time | 1.87 s/image |
| Successful pairs | 32/40 |
| Ambiguous/failure pairs | 8/40 |

### Failure Analysis

The 8 failure cases are intentional marker-free periodic-array cases.
Multiple candidate locations produce essentially identical NCC scores.

For these cases:

- Ambiguity ratio = 1.000
- Confidence = LOW
- The matcher reports multiple candidate locations instead of falsely claiming a unique location.

The remaining 32 marker-aided cases achieve sub-pixel localization in the reproduced
evaluation, with a median error of 0.45 px.

## Results Summary

**Accuracy Metrics (40-pair test suite):**
- Accuracy within 1–50 pixels: **80.0%** (32/40 pairs correct)
- Mean pixel error: **65.90 px** on the jittered dataset
- Median pixel error: **0.50 px** (high-performing cases)

**RGB test check:**
- 3 representative RGB pairs were evaluated and all landed within a fraction of a pixel of ground truth, confirming grayscale conversion inside the NCC matcher is working correctly.

**Jittered dataset check:**
- Re-generated the full 40-pair dataset with ±10–20% random size perturbations.
- Accuracy remained **80.0%** across all tolerances, with the same 8 intentionally ambiguous LOW-confidence cases and 32 HIGH-confidence successful cases.

**Confidence Detection:**
- **8 FAILED pairs** (periodic ambiguity, no navigation marker):
  - Ambiguity ratio: exactly **1.000** (tied second-best locations)
  - Confidence label: **LOW**
  - Pixel error range: 225–786 px
  
- **32 PASSED pairs** (marker-aided localization):
  - Ambiguity ratio: **0.980–0.990** (clear winner)
  - Confidence label: **HIGH**
  - Pixel error: typically < 1 px

**Agreement with ground truth:** 100% (confidence label perfectly predicts success/failure)


---

## Known Limitations

### Periodic Ambiguity Without Disambiguating Features

The matcher uses template correlation, which fundamentally cannot distinguish between identical or nearly-identical repeating patterns without additional context. The test suite includes **8 pairs with no navigation marker** (intentional hard cases):

- **DRAM checkerboard (4 pairs)**: Grid of identical cells. Without marker, multiple grid locations have equally high correlation.
- **FinFET stripes (4 pairs)**: Parallel lines. Multiple stripe boundaries produce equally high correlation.

**Result:** The matcher correctly assigns **ratio = 1.000 (LOW confidence)** to these cases, flagging them as ambiguous rather than returning a spurious location.

**Industry context:** Real SEM workflows address this by:
1. Printing a navigation fiducial (marker) on the chip
2. Using multi-modal cues (e.g., overlay with charged regions or defects)
3. Applying multi-hypothesis tracking across drift measurements

This project simulates the marker-aided case (32 pairs) and the marker-free case (8 pairs) to demonstrate both the success and failure modes.

---

## Project Structure

```
patch-localization/
├── algorithm/
│   ├── core/                          # Main inference and evaluation
│   │   ├── infer.py                   # Single-pair inference entry point
│   │   ├── ncc.py                     # NCC template matching (cv2.matchTemplate)
│   │   ├── batch_eval.py              # Batch evaluation over 40 pairs
│   │   ├── rotation_search.py         # Image rotation helper
│   │   ├── scale_rotation_search.py   # Scale and rotation helper
│   │   ├── plot_pr_curve.py           # Accuracy-vs-tolerance plot
│   │   ├── test_all_pairs.py          # Test rotation search
│   │   ├── test_rotation_all.py       # Test rotation on all pairs
│   │   ├── preprocessing.py           # Image preprocessing utilities
│   │   └── batch_results.csv          # Batch evaluation results
│   │
│   ├── dataset/                       # Dataset generation
│   │   ├── generate_dataset.py        # Generate 40 synthetic pairs
│   │   └── verify_gt.py               # Verify ground-truth annotations
│   │
│   ├── scripts/                       # Utility and debug scripts
│   │   ├── debug_check.py             # Debug NCC matching
│   │   ├── check_hard_cases.py        # Analyze failed pairs
│   │   └── verify_gt_point.py         # Visualize ground-truth locations
│   │
│   ├── tests/                         # Test data
│   │   └── synthetic_data/
│   │       ├── pair_*.png             # Generated reference/search images
│   │       └── ground_truth.csv       # Locations and metadata
│   │
│   ├── docs/                          # Documentation
│   │   └── examples/                  # Visual evidence
│   │       ├── gt_check_pair_0_*.png  # Example failure case (periodic ambiguity)
│   │       └── gt_check_pair_2_*.png  # Example success case (marker-aided)
│   │
│   └── configs/                       # Configuration placeholders
│
├── requirements.txt                   # Python package dependencies
├── citations.md                       # References and citations
└── README.md                          # This file
```

---

## Implementation Details

### NCC Template Matching (core/ncc.py)

Uses OpenCV's `cv2.matchTemplate()` with the `cv2.TM_CCOEFF_NORMED` norm (normalized correlation coefficient). Returns a score in [−1, 1], where 1 is perfect match.

### Multi-Scale and Multi-Rotation Search (core/infer.py)

- **Rotation range:** ±3° in 1° steps (7 angles total)
- **Scale range:** 0.08–0.12 in 0.005 steps (~8 scales total)
- **Tie-breaking:** When multiple angles/scales yield near-identical top scores (within 0.005), selects the match closest to the image center (minimizing predicted drift).

### Ambiguity Detection (core/infer.py)

1. Collect all candidate matches from the multi-scale/rotation search
2. Cluster candidates by spatial proximity (20-pixel grid)
3. Keep the best score in each cluster
4. Compute ratio: `score_2nd_best_cluster / score_best_cluster`
5. If ratio ≥ 0.995 → **LOW confidence** (ambiguous)
6. If ratio < 0.995 → **HIGH confidence** (confident match)

Threshold **0.995** was calibrated against the 40-pair test suite and achieves 100% accuracy in predicting success/failure.

---

## Training and Model Weights

**This project contains no neural network models, training scripts, or model weights.** The approach is purely classical template matching with hand-crafted augmentation. This is acceptable under the problem statement, which permits non-DL solutions.

---

## References

See [citations.md](citations.md) for full citations and their justifications.

---

## Submission Info

**Team:** Patch Localization Team  
**Challenge:** Applied Materials Navigation-Error Recovery (SEMI-India Hackathon)  
**Deadline:** August 16, 2025  
**Approach:** Classical CV (NCC + multi-scale/rotation) + confidence detection (Lowe-inspired ratio test)  
**Novelty:** Ambiguity-detection via ratio test addresses the periodic-pattern hard case systematically.

---

## License

 MIT

---

## Questions or Issues?

For questions about setup, usage, or results, please refer to the code comments or open an issue in the repository.
