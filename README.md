# Patch Localization

## Project Overview

Classical computer vision for locating a reference crop in a larger SEM-style
search image. The implementation uses OpenCV normalized cross-correlation,
multi-scale search, multi-rotation search, and phase-correlation reranking.
It does not use machine learning or neural-network weights.

## Phase 2 Updates (from Phase 1)

Phase 2 extends the Phase 1 NCC matcher to handle pose variation, repeated
structures, degraded images, and explicit match rejection:

- `algorithm/core/scale_rotation_search.py` searches the direct zoom range
  8.0-12.0 and converts each zoom to the internal template scale used by NCC.
- `algorithm/core/infer.py` adds coarse-to-fine multi-angle and multi-zoom
  search, keeps multiple peaks per scale, clusters candidates by location, and
  computes a top-two margin for ambiguity and rejection decisions. It also
  supports weighted-NCC, phase-correlation, subregion-consistency, and
  majority-vote reranking helpers.
- `register.py` is the Phase 2 batch entry point. It accepts several common
  reference/search CSV column names, resolves relative image paths from the
  CSV directory, runs each pair in a timed worker, reranks the strongest
  candidates, converts internal scale to output zoom, and writes both
  predictions and a companion timings/error log. A `top2_margin` threshold
  (`0.0002`) controls the `found` decision.
- `register.py` uses a two-stage runtime search: a half-resolution grid pass
  covers the full angle/zoom range to cheaply rank candidates, followed by
  full-resolution NCC refinement of only the top 5 candidates before
  phase-correlation reranking. This reduced median runtime from approximately
  5.07 seconds to approximately 2.4 seconds per pair.
- `analyze_scores.py` joins `predictions_timings.csv` (`best_score`,
  `top2_margin`, and `found`) with `ground_truth.csv` to compare GT-present and
  GT-absent score distributions and calibrate `SCORE_THRESHOLD`.
- `algorithm/dataset/generate_dataset_phase2.py` adds nominal, degraded, and
  absent synthetic pairs, grayscale and RGB generation, scale/size jitter,
  Gaussian and shot noise, blur, and rotated references.
- `algorithm/tests/official_phase2/` adds a self-constructed 20-pair validation set mimicking the four organizer categories (A/B/C/D), used for local sanity-checking against the I/O contract,
  ground truth, manifest, reference/search images, predictions, and timing
  files. The repository's **self-constructed local fixture** contains Set A = 8 pairs, Set B = 6 pairs,
  Set C = 4 pairs, and Set D = 2 pairs. The generated Phase 2 data is retained
  separately as development data in `algorithm/tests/synthetic_data_phase2/`,
  `algorithm/tests/synthetic_data_phase2_v2/`, and
  `algorithm/tests/synthetic_data_phase2_set_d/` (20 RGB Set-D-style pairs).

The official scorer is `score_official_phase2.py`; it reports per-set credit,
rejection F1, pose accuracy, and runtime from the official Phase 2 fixtures.

## Setup

```bash
git clone https://github.com/varrshaa10/patch-localization.git
cd patch-localization
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Single-Pair Inference

```bash
python algorithm/core/infer.py \
  --reference path/to/reference.png \
  --search path/to/search.png
```

`infer.py` searches the configured angle and scale ranges, extracts candidate
peaks, and reranks the strongest candidates with phase correlation. The
reported result includes the predicted center, score, angle, and scale.

## Batch Registration

The repository also contains 100 generated Phase 2 synthetic pairs with
nominal, degraded, absent, grayscale, and RGB variants. They are retained as
development data and are not part of the documented 20-pair evaluation below.
The repository's batch entry point accepts a CSV containing `pair_id`,
`reference_path` (or `ref_path`), and `search_path` columns:

```bash
python register.py \
  --input algorithm/tests/official_phase2/pairs.csv \
  --output algorithm/tests/official_phase2/predictions.csv
```

The output columns are `pair_id`, `x`, `y`, `theta`, `scale`, `found`, and
`score`. A companion timings CSV records runtime, margin, status, and errors.

### Detection Decision

Registration uses the margin and best-score thresholds below:

```text
FIND_THRESHOLD = 0.0002
SCORE_THRESHOLD = 0.30
found = 1 when top2_margin > FIND_THRESHOLD AND best_score > SCORE_THRESHOLD
found = 0 otherwise
```

## Evaluation And Submission

The documented evaluation uses the checked-in 20-pair fixture:

`algorithm/tests/official_phase2/pairs.csv` contains 20 local fixture pairs:
Set A = 8, Set B = 6, Set C = 4, and Set D = 2. It is useful for checking the
expected reference/search CSV format and the local scorer:

```bash
python register.py \
  --input algorithm/tests/official_phase2/pairs.csv \
  --output algorithm/tests/official_phase2/predictions.csv
python score_official_phase2.py
```

Here `score_official_phase2.py` reads the resulting `predictions.csv` and
`predictions_timings.csv`, together with the checked-in ground truth. The
fixture is self-constructed in this repository; it is not confirmed
organizer-provided data and must not be presented as organizer validation.

### Actual Organizer Submission

For the real evaluation, use the reference/search pairs and submission format
provided separately by the organizers. Run `register.py` on that supplied
input and submit the resulting predictions file exactly as requested. The
organizer dataset is not present in this repository, so neither the local
20-pair fixture nor our generated 100-pair dataset is the actual submission
test set.

The official-style scorer is `score_official_phase2.py`. Its latest local
fixture result is shown below. Set C is scored via rejection F1, not
localization credit.

| Metric | Result | Organizer baseline |
|---|---:|---:|
| Set A mean credit | 0.975 | 1.000 |
| Set B mean credit | 0.967 | 0.467 |
| Set C rejection (TN out of pairs) | 3/4 correctly rejected | — |
| Set D mean credit | 1.000 | 1.000 |
| Rejection F1 | 0.9697 (TP=16, FP=1, FN=0, TN=3) | 0.897 |
| Scale error, median / worst | 1.004% / 3.030% | 1.0% / 3.0% |
| Theta error, median / worst | 0.300° / 0.900° | 0.35° / 1.10° |
| Median runtime | ~2.4s | — |

To reproduce the report after generating official predictions:

```bash
python score_official_phase2.py
```

## Running Phase 2 (for organizers)

From a clean machine, clone the repository, create and activate a virtual
environment, install the pinned dependencies, run registration on the checked-in
official pairs, and score the resulting predictions:

```bash
git clone https://github.com/varrshaa10/patch-localization.git
cd patch-localization
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
python register.py --input algorithm/tests/official_phase2/pairs.csv --output algorithm/tests/official_phase2/predictions.csv
python score_official_phase2.py
```

`register.py` also writes
`algorithm/tests/official_phase2/predictions_timings.csv`. The scorer reads
these two output files plus the checked-in `ground_truth.csv` and reports the
Phase 2 evaluation metrics.

## Repository Contents

```text
patch-localization/
├── algorithm/
│   ├── core/
│   │   ├── infer.py
│   │   ├── ncc.py
│   │   └── scale_rotation_search.py
│   └── dataset/
│       └── generate_dataset_phase2.py
├── register.py
├── analyze_scores.py
├── requirements.txt
└── README.md
```

## Limitations

Periodic structures can produce several nearly equivalent correlation peaks.
The margin-based decision helps reject ambiguous matches, but it cannot create
information that is absent from the images. Navigation markers, additional
modalities, or temporal context are needed to disambiguate genuinely repeating
patterns. The present and absent score distributions also overlap (present min
`0.338`, absent max `0.377` on the local 20-pair set), which is a known,
disclosed limitation of the rejection threshold and is not fully resolved.
