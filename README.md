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
  Set C = 4 pairs, and Set D = 2 pairs. The generated Phase 2 data is kept in
  `algorithm/tests/synthetic_data_phase2/` (older 80-pair run),
  `algorithm/tests/synthetic_data_phase2_v2/` (current 100-pair evaluation), and
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
SCORE_THRESHOLD = 0.30
found = 1 when top2_margin > FIND_THRESHOLD AND best_score > SCORE_THRESHOLD
found = 0 otherwise
score = best_score in both the found=1 and found=0 cases
```

The `SCORE_THRESHOLD` was added because `top2_margin` alone was insufficient
to reject absent-reference pairs: a low-quality best match can still clearly
beat the second-best candidate. It was calibrated with `analyze_scores.py`
against a self-generated present/absent score distribution (GT-present range
`0.338-0.954`, GT-absent range `0.258-0.377`, with some overlap); `0.30` was
chosen to preserve recall on present pairs because localization+pose scoring
(60 pts) outweighs rejection scoring (15 pts). The margin and best score are
written to the timings CSV, while failed workers and timeouts produce a zero
result and are recorded in the timing log.

## Phase 2 Synthetic Dataset

The Phase 2 generator creates nominal, degraded, and absent examples with
grayscale or RGB variants:

```bash
python algorithm/dataset/generate_dataset_phase2.py \
  --output_dir algorithm/tests/synthetic_data_phase2_v2 \
  --num_nominal 30 \
  --num_degraded 30 \
  --num_absent 40 \
  --seed 12345
```

Run registration on the generated pairs:

```bash
python register.py \
  --input algorithm/tests/synthetic_data_phase2_v2/ground_truth_combined.csv \
  --output outputs/final_main_predictions_v2.csv
```

The current local evaluation target is the 100-pair `v2` dataset: 30 nominal
present pairs, 30 degraded present pairs, and 40 absent pairs. Evaluate the
predictions against `ground_truth_combined.csv`; report localization on the 60
present pairs and rejection metrics on all 100 pairs. The generator writes each
manifest as `ground_truth_combined.csv` alongside its `pair_<id>/` directories.
The documented `register.py` evaluation produced 52 TP, 25 FP, 8 FN, and
15 TN: precision `0.675325`, recall `0.866667`, and rejection F1 `0.759124`.
It localized **26/60 present pairs within 5 px**. The separate experimental
`tmp_eval_v2.py` evaluator reported 56 TP, 22 FP, 4 FN, and 18 TN because it
uses a different search and scoring path; that result is not the score for the
README registration command. The older 80-pair result (**24/60** or **25/60**)
was also from an earlier search version and is not directly comparable.
Existing checked-in manifests were created with an older path format; regenerate
the selected directory before running registration so the manifest paths match
its local pair directories.

This synthetic evaluation is only for local algorithm development. It is not
the organizer submission dataset. The `v2` directory is a later local variant
with 40 absent pairs, while `synthetic_data_phase2_set_d` is a separate 20-pair
RGB experiment; neither replaces the official fixture.

## Official Phase 2 Evaluation

The official scorer is `score_official_phase2.py`. With the 20-pair official
predictions, the latest verified results were:

For the local organizer-format sanity check, run registration on the
checked-in self-constructed input `algorithm/tests/official_phase2/pairs.csv`.
This input points to 20 local fixture image pairs split across Sets A-D. The resulting
`predictions.csv` and `predictions_timings.csv` are the files consumed by the
local scorer. This fixture is **not confirmed organizer-provided data** and
must not be treated as the real submission test set.

The generated datasets are also required for local development: run the
100-pair `algorithm/tests/synthetic_data_phase2_v2/` evaluation to measure
behavior on our controlled nominal, degraded, and absent cases. The 80-pair
dataset and 20-pair RGB Set-D-style dataset are older or specialized local
experiments. They do not replace organizer data, and their scores are not
organizer scores.

For the actual organizer evaluation, use the dataset and submission format
provided by the organizers. Run `register.py` on their supplied reference/search
pairs and submit the resulting predictions file exactly as requested. The
organizer's hidden or supplied test data is not present in this repository, so
the local 20-pair fixture and our generated 100-pair dataset are not sufficient
to claim organizer validation by themselves.

Set C is scored via the Rejection F1 metric below, not localization credit.

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
