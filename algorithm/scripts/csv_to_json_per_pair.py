#!/usr/bin/env python3
"""Convert each per-pair CSV metadata file into JSON and remove the CSV.

The per-pair JSON intentionally excludes ref_path/search_path because each pair is
already stored in its own folder as reference.png and search.png; those paths are
not needed for browsing or review.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "synthetic_data"


def parse_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def main():
    pair_dirs = sorted(DATA_DIR.glob("pair_*"))
    converted = 0

    for pair_dir in pair_dirs:
        if not pair_dir.is_dir():
            continue
        csv_path = pair_dir / "ground_truth.csv"
        if not csv_path.exists():
            continue

        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No rows found in {csv_path}")

        row = rows[0]
        clean_row = {
            "pair_id": parse_value(row.get("pair_id")),
            "architecture": row.get("architecture"),
            "gt_x": parse_value(row.get("gt_x")),
            "gt_y": parse_value(row.get("gt_y")),
            "angle": parse_value(row.get("angle")),
            "scale": parse_value(row.get("scale")),
            "add_marker": parse_value(row.get("add_marker")),
            "mode": row.get("mode"),
        }

        json_path = pair_dir / "ground_truth.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(clean_row, f, indent=2)
            f.write("\n")

        csv_path.unlink()
        converted += 1
        print(f"Converted {pair_dir.name}: {json_path.name}")

    print(f"Converted {converted} pair folders to ground_truth.json")


if __name__ == "__main__":
    main()
