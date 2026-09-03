#!/usr/bin/env python3
"""One-time migration for the synthetic dataset layout.

Moves the flat layout used by older scripts into per-pair subfolders while
retaining a canonical combined CSV at the root.
"""

import csv
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "synthetic_data"
LEGACY_CSV = DATA_DIR / "ground_truth.csv"
COMBINED_CSV = DATA_DIR / "ground_truth_combined.csv"


def find_matching_file(directory: Path, stem: str) -> Path | None:
    """Return the first file whose name contains the requested stem."""
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and stem in candidate.name:
            return candidate
    return None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LEGACY_CSV.exists():
        raise FileNotFoundError(f"Expected combined CSV at {LEGACY_CSV}")

    with LEGACY_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {LEGACY_CSV}")

    fieldnames = rows[0].keys()
    shutil.copy2(LEGACY_CSV, COMBINED_CSV)

    for row in rows:
        pair_id = row["pair_id"]
        pair_dir = DATA_DIR / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        ref_candidate = find_matching_file(DATA_DIR, f"pair_{pair_id}_reference")
        search_candidate = find_matching_file(DATA_DIR, f"pair_{pair_id}_search")

        if ref_candidate is None:
            raise FileNotFoundError(f"Missing reference image for pair {pair_id}")
        if search_candidate is None:
            raise FileNotFoundError(f"Missing search image for pair {pair_id}")

        target_ref = pair_dir / "reference.png"
        target_search = pair_dir / "search.png"
        shutil.copy2(ref_candidate, target_ref)
        shutil.copy2(search_candidate, target_search)

        per_pair_csv = pair_dir / "ground_truth.csv"
        with per_pair_csv.open("w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerow(row)

    print(f"Migrated {len(rows)} pairs into {DATA_DIR}")
    print(f"Combined CSV preserved at {COMBINED_CSV}")


if __name__ == "__main__":
    main()
