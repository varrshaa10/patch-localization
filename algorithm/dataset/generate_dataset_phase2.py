"""Generate the Applied Materials Phase 2 synthetic localization dataset."""
import argparse
import csv
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from generate_dataset import draw_dram, draw_finfet


REF_SIZE = 1000
SEARCH_SIZE = 1000
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tests", "synthetic_data_phase2")
)


DEGRADED_LEVELS = (
    {"name": "mild", "gaussian_std": 8, "shot_scale": 0.30, "blur": 0.9},
    {"name": "moderate", "gaussian_std": 14, "shot_scale": 0.18, "blur": 1.5},
    {"name": "severe", "gaussian_std": 22, "shot_scale": 0.10, "blur": 2.2},
)


def build_canvas(architecture, size, jitter=False):
    """Build one architecture region at the requested physical resolution."""
    image = Image.new("L", (size, size), color=30)
    draw = ImageDraw.Draw(image)
    if architecture == "dram":
        line_width = 60
        dot_radius = 45
        if jitter:
            line_width = int(line_width * random.uniform(0.80, 1.20))
            dot_radius = int(dot_radius * random.uniform(0.80, 1.20))
        draw_dram(draw, size, line_w=line_width, dot_r=dot_radius)
    else:
        fin_width = 50
        if jitter:
            fin_width = int(fin_width * random.uniform(0.80, 1.20))
        draw_finfet(draw, size, fin_w=fin_width)
    return image


def add_marker(image, center_x, center_y):
    draw = ImageDraw.Draw(image)
    draw.line(
        [(center_x - 120, center_y - 120), (center_x + 120, center_y + 120)],
        fill=255,
        width=25,
    )
    draw.line(
        [(center_x - 120, center_y + 120), (center_x + 120, center_y - 120)],
        fill=0,
        width=25,
    )


def process_image(image, gaussian_std, shot_scale, blur):
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
    array = np.asarray(image, dtype=np.float32)
    shot = np.random.poisson(np.maximum(array, 0) * shot_scale) / shot_scale
    noise = np.random.normal(0, gaussian_std, array.shape)
    return Image.fromarray(np.clip(shot + noise, 0, 255).astype(np.uint8))


def process_rgb_image(image, gaussian_std, shot_scale, blur):
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
    array = np.asarray(image, dtype=np.float32)
    processed = np.zeros_like(array)
    for ch in range(3):
        channel = array[:, :, ch]
        shot = np.random.poisson(np.maximum(channel, 0) * shot_scale) / shot_scale
        noise = np.random.normal(0, gaussian_std, channel.shape)
        processed[:, :, ch] = np.clip(shot + noise, 0, 255)
    return Image.fromarray(processed.astype(np.uint8), mode="RGB")


def build_rgb_canvas(architecture, size, jitter=False):
    image = Image.new("RGB", (size, size), color=(20, 20, 25))
    draw = ImageDraw.Draw(image)
    if architecture == "dram":
        line_width = 60
        dot_radius = 45
        if jitter:
            line_width = int(line_width * random.uniform(0.80, 1.20))
            dot_radius = int(dot_radius * random.uniform(0.80, 1.20))
        tint = (110, 130, 170)
        draw_dram(draw, size, line_w=line_width, dot_r=dot_radius, mode="RGB", tint=tint)
    else:
        fin_width = 50
        if jitter:
            fin_width = int(fin_width * random.uniform(0.80, 1.20))
        tint = (100, 170, 130)
        draw_finfet(draw, size, fin_w=fin_width, mode="RGB", tint=tint)
    return image


def generate_set_d_pair(architecture, pair_id, output_dir, variant_offset=0):
    scale = random.uniform(8.0, 12.0)
    canvas_size = int(round(SEARCH_SIZE * scale))
    canvas = build_rgb_canvas(architecture, canvas_size, jitter=False)

    margin = REF_SIZE // 2 + 10
    center_x = random.randint(margin, canvas_size - margin)
    center_y = random.randint(margin, canvas_size - margin)
    draw = ImageDraw.Draw(canvas)
    tint = (255, 255, 255) if (pair_id + variant_offset) % 2 == 0 else (200, 200, 220)
    draw.line([(center_x - 120, center_y - 120), (center_x + 120, center_y + 120)], fill=tint, width=25)
    draw.line([(center_x - 120, center_y + 120), (center_x + 120, center_y - 120)], fill=(0, 0, 0), width=25)

    reference_source = canvas.copy()
    reference = reference_source.crop(
        (center_x - REF_SIZE // 2, center_y - REF_SIZE // 2,
         center_x + REF_SIZE // 2, center_y + REF_SIZE // 2)
    )
    theta = random.uniform(-5.0, 5.0)
    reference = reference.rotate(theta, resample=Image.Resampling.BICUBIC, fillcolor=(30, 30, 30))
    reference = process_rgb_image(reference, gaussian_std=4, shot_scale=0.8, blur=0.6)
    search = process_rgb_image(canvas.resize((SEARCH_SIZE, SEARCH_SIZE), Image.Resampling.BOX), gaussian_std=6, shot_scale=0.4, blur=0.5)

    pair_dir = os.path.join(output_dir, f"pair_{pair_id}")
    os.makedirs(pair_dir, exist_ok=True)
    reference_path = os.path.join(pair_dir, "reference.png")
    search_path = os.path.join(pair_dir, "search.png")
    reference.save(reference_path)
    search.save(search_path)

    return {
        "pair_id": pair_id,
        "architecture": architecture,
        "category": "present",
        "ref_path": f"pair_{pair_id}/reference.png",
        "search_path": f"pair_{pair_id}/search.png",
        "gt_x": round(center_x / scale, 2),
        "gt_y": round(center_y / scale, 2),
        "gt_theta": round(theta, 2),
        "gt_scale": round(scale, 4),
        "found": 1,
        "severity": "set_d_present",
        "mode": "RGB",
    }


def generate_pair(architecture, pair_id, category, output_dir, severity=None):
    scale = random.uniform(8.0, 12.0)
    canvas_size = int(round(SEARCH_SIZE * scale))
    jitter = category == "present_degraded"
    canvas = build_canvas(architecture, canvas_size, jitter=jitter)

    margin = REF_SIZE // 2 + 10
    center_x = random.randint(margin, canvas_size - margin)
    center_y = random.randint(margin, canvas_size - margin)
    if category != "absent":
        add_marker(canvas, center_x, center_y)
    reference_source = canvas.copy()
    if category == "absent":
        add_marker(reference_source, center_x, center_y)

    reference = reference_source.crop(
        (center_x - REF_SIZE // 2, center_y - REF_SIZE // 2,
         center_x + REF_SIZE // 2, center_y + REF_SIZE // 2)
    )
    theta = random.uniform(-5.0, 5.0)
    reference = reference.rotate(theta, resample=Image.Resampling.BICUBIC, fillcolor=30)

    if category == "present_nominal":
        ref_params = {"gaussian_std": 4, "shot_scale": 0.8, "blur": 0.6}
        search_params = {"gaussian_std": 6, "shot_scale": 0.4, "blur": 0.5}
        search_source = canvas
    elif category == "present_degraded":
        ref_params = {"gaussian_std": severity["gaussian_std"], "shot_scale": severity["shot_scale"], "blur": severity["blur"]}
        search_params = {"gaussian_std": severity["gaussian_std"], "shot_scale": severity["shot_scale"], "blur": severity["blur"]}
        search_source = canvas
    else:
        ref_params = {"gaussian_std": 4, "shot_scale": 0.8, "blur": 0.6}
        search_params = {"gaussian_std": 6, "shot_scale": 0.4, "blur": 0.5}
        search_source = build_canvas(architecture, canvas_size, jitter=False)

    reference = process_image(reference, **ref_params)
    search = process_image(search_source.resize((SEARCH_SIZE, SEARCH_SIZE), Image.Resampling.BOX), **search_params)

    pair_dir = os.path.join(output_dir, f"pair_{pair_id}")
    os.makedirs(pair_dir, exist_ok=True)
    reference_path = os.path.join(pair_dir, "reference.png")
    search_path = os.path.join(pair_dir, "search.png")
    reference.save(reference_path)
    search.save(search_path)

    found = 0 if category == "absent" else 1
    return {
        "pair_id": pair_id,
        "architecture": architecture,
        "category": category,
        "ref_path": f"pair_{pair_id}/reference.png",
        "search_path": f"pair_{pair_id}/search.png",
        "gt_x": round(center_x / scale, 2) if found else None,
        "gt_y": round(center_y / scale, 2) if found else None,
        "gt_theta": round(theta, 2) if found else None,
        "gt_scale": round(scale, 4) if found else None,
        "found": found,
        "severity": severity["name"] if severity else "nominal",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_nominal", type=int, default=30)
    parser.add_argument("--num_degraded", type=int, default=30)
    parser.add_argument("--num_absent", type=int, default=20)
    parser.add_argument("--architecture", choices=["dram", "finfet", "both"], default="both")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    architectures = ["dram", "finfet"] if args.architecture == "both" else [args.architecture]
    requests = (["present_nominal"] * args.num_nominal +
                ["present_degraded"] * args.num_degraded +
                ["absent"] * args.num_absent)
    rows = []
    for pair_id, category in enumerate(requests):
        architecture = architectures[pair_id % len(architectures)]
        severity = (DEGRADED_LEVELS[pair_id % len(DEGRADED_LEVELS)]
                    if category == "present_degraded" else None)
        rows.append(generate_pair(architecture, pair_id, category, args.output_dir, severity))

    fields = ["pair_id", "architecture", "category", "ref_path", "search_path",
              "gt_x", "gt_y", "gt_theta", "gt_scale", "found"]
    combined_path = os.path.join(args.output_dir, "ground_truth_combined.csv")
    with open(combined_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    for row in rows:
        json_path = os.path.join(args.output_dir, f"pair_{row['pair_id']}", "ground_truth.json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump({field: row[field] for field in fields}, json_file, indent=2)
            json_file.write("\n")

    counts = {category: requests.count(category) for category in ("present_nominal", "present_degraded", "absent")}
    print(f"Generated {len(rows)} pairs in {args.output_dir}")
    print("Counts: " + ", ".join(f"{category}={count}" for category, count in counts.items()))
    print(f"Ground truth: {combined_path}")


if __name__ == "__main__":
    main()