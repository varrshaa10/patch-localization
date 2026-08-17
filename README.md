# Patch Localization: Classical NCC-Based SEM Site Relocalization

## Project Overview

**Patch-Localization** is a classical computer vision project (no machine learning, no neural networks, no training) that solves the **drift-sense site relocalization problem**: given a reference image crop (patch) of a semiconductor microstructure and a larger, noisier search image, find where the reference is located within the search image.

The solution uses **normalized cross-correlation (NCC)** template matching combined with multi-scale and multi-rotation search, augmented with an **ambiguity-ratio confidence metric** inspired by SIFT's ratio test. This allows detection of inherently ambiguous cases (periodic patterns without disambiguating features) without training-based approaches.

**Key characteristics:**
- Classical computer vision approach (no DL models, training, or model weights)
- Synthetic dataset: 60 image pairs (30 DRAM + 30 FinFET microstructures), with per-shape jitter enabled
- Accuracy: **83.3%** within 1–20 px and **85.0%** at 50 px tolerance
- Confidence detection: fixed ambiguity-ratio rule remains the strongest practical decision rule
- Includes about 10 intentionally hard/no-marker cases representing periodic ambiguity

---

## Quick Start (pipeline)

From the `algorithm/` directory, the one-command repro path is:

```bash
python pipeline.py --config config.yaml --skip-generate
```

This runs the end-to-end workflow using the existing dataset, executes batch evaluation, and saves the accuracy plot without regenerating the synthetic data first.

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
- `matplotlib` — Plotting and PR-curve generation
- `PyYAML` — YAML config parsing
- `pandas` — CSV analysis and evaluation logs
- `joblib` — Saving/loading the optional calibrator model
- `scikit-learn` — Optional calibration experiment

---

## Configuration

The project now keeps the core matching and evaluation settings in `algorithm/config.yaml`. This file stores the synthetic-data generation settings, matching ranges, ambiguity threshold, and evaluation tolerances used by the pipeline and associated scripts. The pipeline reads the dataset generation parameters from this config file, while the matching logic and batch-eval summaries still use the same core algorithm behavior.

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
python dataset/generate_dataset.py --architecture both --num_pairs 60 --output_dir ./tests/synthetic_data
python dataset/generate_dataset.py --architecture both --num_pairs 5 --output_dir ./tests/synthetic_data_rgb --color
python dataset/generate_dataset.py --architecture both --num_pairs 60 --output_dir ./tests/synthetic_data --jitter
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

This evaluates the matcher on all 60 ground-truth pairs, computes per-pair pixel error, and generates a summary report.

**Output:**
```
Accuracy within 1px: 83.3%
Accuracy within 2px: 83.3%
...
Accuracy within 50px: 85.0%

Ambiguity ratio distribution:
Ratio on FAILED pairs (n=10): min=0.999, max=1.000, mean=1.000
Ratio on PASSED pairs (n=50): min=0.980, max=0.993, mean=0.985
```

Results are written to `batch_results.csv`.

---
### Ground Truth

Each generated dataset includes a `ground_truth.csv` file containing the metadata required for quantitative evaluation.

The ground-truth file records:

| Field | Description |
|---|---|
| `pair_id` | Unique image-pair identifier |
| `architecture` | Semiconductor pattern type: DRAM or FinFET |
| `ref_path` | Path to the reference image |
| `search_path` | Path to the search image |
| `gt_x` | Ground-truth x-coordinate of the reference pattern center |
| `gt_y` | Ground-truth y-coordinate of the reference pattern center |
| `angle` | Applied rotation |
| `scale` | Applied scale factor |
| `add_marker` | Indicates whether the distinguishing marker is present |
| `mode` | Image mode, such as grayscale (`L`) or RGB |

The `gt_x` and `gt_y` coordinates are used to calculate localization error and accuracy during evaluation.

The `add_marker` field records whether the distinguishing marker was included in the generated sample and is available for analysis of marker-present and marker-absent cases.

---
## Results Summary

**Accuracy Metrics (60-pair test suite):**
- Accuracy within **1–20 px**: **83.3%** (50/60 pairs correct)
- Accuracy within **50 px**: **85.0%** (51/60 pairs correct)
- Mean pixel error: **72.22 px** on the jittered 60-pair dataset
- Median pixel error: **0.42 px** (high-performing cases)

**Hard and ambiguous cases:**
- The current 60-pair suite contains about **10 hard/no-marker pairs** that are intentionally periodic and ambiguous.
- These cases cluster near an ambiguity ratio of **1.000** and are correctly flagged as LOW confidence instead of returning a misleading location.

**RGB test check:**
- 3 representative RGB pairs were evaluated and all landed within a fraction of a pixel of ground truth, confirming grayscale conversion inside the NCC matcher is working correctly.

**Jittered dataset check:**
- Re-generated the full 60-pair dataset with per-shape jitter enabled.
- Accuracy remained **83.3%** across the 1–20 px tolerance band, and **85.0%** at 50 px, with roughly 10 intentionally ambiguous LOW-confidence cases and 50 successful HIGH-confidence cases.

**Confidence Detection:**
- **10 FAILED pairs** (periodic ambiguity, no navigation marker):
  - Ambiguity ratio: **0.999–1.000** (near-tied second-best locations)
  - Confidence label: **LOW**
  - Pixel error range: 136–695 px
  
- **50 PASSED pairs** (marker-aided localization):
  - Ambiguity ratio: **0.980–0.993** (clear winner)
  - Confidence label: **HIGH**
  - Pixel error: typically < 1 px

**Agreement with ground truth:** the fixed ambiguity-ratio rule remains the strongest real-world decision signal for this dataset.

### Bonus: Learned Confidence Calibrator (Optional)

The `algorithm/train_calibrator.py` script is a comparison experiment only. It trains a logistic-regression model on the collected `ncc_score` and `ambiguity_ratio` features, using the labeled examples in `algorithm/training_data/run1.csv` through `run4.csv`.

This data was generated by a repeated loop: regenerate the dataset, run batch evaluation, then copy the per-pair CSV output into `training_data/` for the calibrator experiment. The experiment is intentionally documented as a negative result, not as a bug or omission.

The learned model achieved **83.3% test accuracy**, but this was a degenerate outcome: it predicted almost entirely the majority class, with **0% precision and 0% recall for the FAIL class** because the dataset is heavily imbalanced (roughly 16% failures) and the features were not scaled. By contrast, the existing fixed rule using the **0.995 ambiguity-ratio threshold** achieved **98.3% accuracy on the same test split** and correctly detected failures. That result is the honest final finding: the fixed threshold outperforms the naive learned calibrator on this dataset.


---

## Known Limitations

### Periodic Ambiguity Without Disambiguating Features

The matcher uses template correlation, which fundamentally cannot distinguish between identical or nearly-identical repeating patterns without additional context. The current 60-pair suite includes about **10 pairs with no navigation marker** (intentional hard cases):

- **DRAM checkerboard (approximately 5 pairs)**: Grid of identical cells. Without marker, multiple grid locations have equally high correlation.
- **FinFET stripes (approximately 5 pairs)**: Parallel lines. Multiple stripe boundaries produce equally high correlation.

**Result:** The matcher correctly assigns **ratio ≈ 1.000 (LOW confidence)** to these cases, flagging them as ambiguous rather than returning a spurious location.

**Industry context:** Real SEM workflows address this by:
1. Printing a navigation fiducial (marker) on the chip
2. Using multi-modal cues (e.g., overlay with charged regions or defects)
3. Applying multi-hypothesis tracking across drift measurements

This project simulates the marker-aided case (50 pairs) and the marker-free case (10 pairs) to demonstrate both the success and failure modes.

---

## Project Structure

```
patch-localization/
├── algorithm/
│   ├── core/                          # Main inference and evaluation
│   │   ├── infer.py                   # Single-pair inference entry point
│   │   ├── ncc.py                     # NCC template matching (cv2.matchTemplate)
│   │   ├── batch_eval.py              # Batch evaluation over 60 pairs
│   │   ├── rotation_search.py         # Image rotation helper
│   │   ├── scale_rotation_search.py   # Scale and rotation helper
│   │   ├── plot_pr_curve.py           # Accuracy-vs-tolerance plot
│   │   ├── test_all_pairs.py          # Test rotation search
│   │   ├── test_rotation_all.py       # Test rotation on all pairs
│   │   ├── preprocessing.py           # Image preprocessing utilities
│   │   └── batch_results.csv          # Batch evaluation results
│   │
│   ├── dataset/                       # Dataset generation
│   │   ├── generate_dataset.py        # Generate 60 synthetic pairs
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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Questions or Issues?

For questions about setup, usage, or results, please refer to the code comments or open an issue in the repository.
