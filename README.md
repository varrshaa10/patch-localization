# Patch Localization

Classical computer vision for locating a reference crop in a larger SEM-style
search image. The implementation uses OpenCV normalized cross-correlation,
multi-scale search, multi-rotation search, and phase-correlation reranking.
It does not use machine learning or neural-network weights.

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