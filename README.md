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
- `algorithm/dataset/generate_dataset_phase2.py` adds nominal, degraded, and
  absent synthetic pairs, grayscale and RGB generation, scale/size jitter,
  Gaussian and shot noise, blur, and rotated references.
- `algorithm/tests/official_phase2/` adds the organizer's 20-pair fixtures,
  ground truth, manifest, reference/search images, predictions, and timing
  files. `algorithm/tests/synthetic_data_phase2/` contains the larger generated
  Phase 2 dataset and registration outputs for local checks.

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

The root-level `register.py` accepts a CSV containing `pair_id`,
`reference_path` (or `ref_path`), and `search_path` columns:

```bash
python register.py \
  --input path/to/pairs.csv \
  --output path/to/predictions.csv
```

The output columns are `pair_id`, `x`, `y`, `theta`, `scale`, `found`, and
`score`. A companion `<output>_timings.csv` records runtime, `top2_margin`,
status, and worker errors.

### Detection Decision

Registration uses the margin returned by the coarse-to-fine search:

```text
FIND_THRESHOLD = 0.0002
found = 1 and score = top2_margin       when top2_margin > FIND_THRESHOLD
found = 0 and score = 1 - top2_margin   otherwise
```

The margin is the signal used for the final `found` decision and is also
written to the timings CSV. Failed workers and timeouts produce a zero result
and are recorded in the timing log.

## Phase 2 Synthetic Dataset

The Phase 2 generator creates nominal, degraded, and absent examples with
grayscale or RGB variants:

```bash
python algorithm/dataset/generate_dataset_phase2.py \
  --output_dir algorithm/tests/synthetic_data_phase2 \
  --num_nominal 30 \
  --num_degraded 30 \
  --num_absent 40 \
  --seed 12345
```

Run registration on the generated pairs:

```bash
python register.py \
  --input algorithm/tests/synthetic_data_phase2/pairs_generated.csv \
  --output outputs/synthetic_margin_check.csv
```

The checked local 80-pair run produced **25/60 present pairs within 5 px** and
detection **F1 = 0.8413**.

## Official Phase 2 Evaluation

The official scorer is `score_official_phase2.py`. With the 20-pair official
predictions, the latest verified results were:

| Metric | Result | Organizer baseline |
|---|---:|---:|
| Set A mean credit | 0.975 | 1.000 |
| Set B mean credit | 0.967 | 0.467 |
| Set D mean credit | 1.000 | 1.000 |
| Rejection F1 | 0.9143 | 0.897 |
| Scale error, median / worst | 1.004% / 3.030% | 1.0% / 3.0% |
| Theta error, median / worst | 0.350° / 0.900° | 0.35° / 1.10° |

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
├── requirements.txt
└── README.md
```

## Limitations

Periodic structures can produce several nearly equivalent correlation peaks.
The margin-based decision helps reject ambiguous matches, but it cannot create
information that is absent from the images. Navigation markers, additional
modalities, or temporal context are needed to disambiguate genuinely repeating
patterns.