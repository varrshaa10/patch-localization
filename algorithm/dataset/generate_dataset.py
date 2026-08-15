"""
Drift-Sense synthetic dataset generator.

Citations for augmentation/noise choices (reuse in Slide 9 / references.md):
1. Shot noise dominance in SEM imaging: Reimer, L. "Scanning Electron Microscopy:
   Physics of Image Formation and Microanalysis", Springer, 1998.
2. Edge/charging brightening effect: Goldstein et al., "Scanning Electron Microscopy
   and X-Ray Microanalysis", Springer, 2018.
3. DRAM/FinFET Manhattan-geometry layouts: Kang & Leblebici, "CMOS Digital
   Integrated Circuits", McGraw-Hill.
"""
import argparse, os, csv, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CANVAS = 10000
REF_SIZE = 1000
SEARCH_SIZE = 1000
DOWNSAMPLE = CANVAS // SEARCH_SIZE  # 10

def draw_dram(draw, size, pitch=150, line_w=60, dot_r=45, mode="L", tint=None, jitter=False):
    """Draw DRAM checkerboard pattern. 
    mode="L": grayscale, mode="RGB": color with optional tint tuple (R, G, B)
    jitter=True: add ±10-20% random variation to shape sizes"""
    if jitter:
        line_w = int(line_w * random.uniform(0.85, 1.15))
        dot_r = int(dot_r * random.uniform(0.85, 1.15))
    for y in range(0, size, pitch):
        fill = tint if tint else 200
        draw.line([(0, y), (size, y)], fill=fill, width=max(1, line_w))
    for x in range(0, size, pitch):
        fill = tint if tint else 200
        draw.line([(x, 0), (x, size)], fill=fill, width=max(1, line_w))
    for y in range(0, size, pitch):
        for x in range(0, size, pitch):
            fill = tint if tint else 255
            r = max(1, dot_r)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=fill)

def draw_finfet(draw, size, pitch=120, fin_w=50, mode="L", tint=None, jitter=False):
    """Draw FinFET stripe pattern.
    mode="L": grayscale, mode="RGB": color with optional tint tuple (R, G, B)
    jitter=True: add ±10-20% random variation to shape sizes"""
    if jitter:
        fin_w = int(fin_w * random.uniform(0.85, 1.15))
    for x in range(0, size, pitch):
        fill = tint if tint else 210
        draw.line([(x, 0), (x, size)], fill=fill, width=max(1, fin_w))
    for gy in (size // 3, 2 * size // 3):
        fill = tint if tint else 140
        draw.rectangle([0, gy - 15, size, gy + 15], fill=fill)

def build_canvas(architecture, mode="L", jitter=False):
    """Build base canvas. mode="L" for grayscale, "RGB" for color.
    jitter=True: add ±10-20% random variation to shape sizes"""
    if mode == "RGB":
        # RGB mode with black background
        img = Image.new("RGB", (CANVAS, CANVAS), color=(30, 30, 30))
    else:
        img = Image.new("L", (CANVAS, CANVAS), color=30)
    draw = ImageDraw.Draw(img)
    
    if architecture == "dram":
        tint = (80, 120, 160) if mode == "RGB" else None  # Blue-gray tint for DRAM
        draw_dram(draw, CANVAS, mode=mode, tint=tint, jitter=jitter)
    else:
        tint = (80, 160, 100) if mode == "RGB" else None  # Green-gray tint for FinFET
        draw_finfet(draw, CANVAS, mode=mode, tint=tint, jitter=jitter)
    return img

def edge_brighten(arr, strength=25):
    gx = np.abs(np.gradient(arr.astype(np.float32), axis=1))
    gy = np.abs(np.gradient(arr.astype(np.float32), axis=0))
    edge_map = np.clip(gx + gy, 0, 255)
    edge_map = edge_map / (edge_map.max() + 1e-6)
    boosted = arr.astype(np.float32) + edge_map * strength
    return np.clip(boosted, 0, 255).astype(np.uint8)

def add_noise(arr, gaussian_std, shot_scale):
    arr_f = arr.astype(np.float32)
    shot = np.random.poisson(arr_f * shot_scale) / max(shot_scale, 1e-6)
    gauss = np.random.normal(0, gaussian_std, arr.shape)
    noisy = shot + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def generate_pair(architecture, pair_id, out_dir, add_marker=True, mode="L", jitter=False):
    """Generate a pair with optional RGB mode and shape jitter.
    mode="L": grayscale (default), mode="RGB": color
    jitter=True: add ±10-20% random variation to shape sizes"""
    canvas = build_canvas(architecture, mode=mode, jitter=jitter)
    draw = ImageDraw.Draw(canvas)

    margin = REF_SIZE // 2 + 10
    cx = random.randint(margin, CANVAS - margin)
    cy = random.randint(margin, CANVAS - margin)

    if add_marker:
        # diagonal cross -- a shape that doesn't already exist anywhere else
        # in the periodic DRAM/FinFET pattern, so it survives NCC discrimination
        # even against a busy, high-contrast background
        marker_color = (255, 255, 255) if mode == "RGB" else 255  # White marker
        marker_color_dark = (0, 0, 0) if mode == "RGB" else 0    # Black for diagonal
        draw.line([(cx - 120, cy - 120), (cx + 120, cy + 120)], fill=marker_color, width=25)
        draw.line([(cx - 120, cy + 120), (cx + 120, cy - 120)], fill=marker_color_dark, width=25)

    ref_crop = canvas.crop((cx - REF_SIZE // 2, cy - REF_SIZE // 2,
                             cx + REF_SIZE // 2, cy + REF_SIZE // 2))
    angle = random.uniform(-3, 3)
    fill_color = (30, 30, 30) if mode == "RGB" else 30
    ref_crop = ref_crop.rotate(angle, resample=Image.BICUBIC, fillcolor=fill_color)
    ref_crop = ref_crop.filter(ImageFilter.GaussianBlur(radius=0.6))
    ref_arr = np.array(ref_crop)
    
    # For RGB, process each channel separately for edge brightening and noise
    if mode == "RGB":
        ref_arr_processed = np.zeros_like(ref_arr)
        for ch in range(3):
            ref_arr_processed[:,:,ch] = edge_brighten(ref_arr[:,:,ch], strength=20)
            ref_arr_processed[:,:,ch] = add_noise(ref_arr_processed[:,:,ch], gaussian_std=4, shot_scale=0.8)
        ref_arr = ref_arr_processed
    else:
        ref_arr = edge_brighten(ref_arr, strength=20)
        ref_arr = add_noise(ref_arr, gaussian_std=4, shot_scale=0.8)

    search_img = canvas.resize((SEARCH_SIZE, SEARCH_SIZE), Image.BOX)
    search_img = search_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    search_arr = np.array(search_img)
    
    if mode == "RGB":
        search_arr_processed = np.zeros_like(search_arr)
        for ch in range(3):
            search_arr_processed[:,:,ch] = edge_brighten(search_arr[:,:,ch], strength=15)
            search_arr_processed[:,:,ch] = add_noise(search_arr_processed[:,:,ch], gaussian_std=6, shot_scale=0.4)
        search_arr = search_arr_processed
    else:
        search_arr = edge_brighten(search_arr, strength=15)
        search_arr = add_noise(search_arr, gaussian_std=6, shot_scale=0.4)

    gt_x = cx / DOWNSAMPLE
    gt_y = cy / DOWNSAMPLE

    os.makedirs(out_dir, exist_ok=True)
    mode_suffix = "_rgb" if mode == "RGB" else ""
    # Store absolute paths for actual saving
    ref_path_abs = os.path.join(out_dir, f"pair_{pair_id}_reference{mode_suffix}.png")
    search_path_abs = os.path.join(out_dir, f"pair_{pair_id}_search{mode_suffix}.png")
    # For CSV, use relative paths from core/ directory
    ref_path_csv = f"../tests/synthetic_data/pair_{pair_id}_reference{mode_suffix}.png"
    search_path_csv = f"../tests/synthetic_data/pair_{pair_id}_search{mode_suffix}.png"
    
    if mode == "RGB":
        Image.fromarray(ref_arr.astype(np.uint8), mode="RGB").save(ref_path_abs)
        Image.fromarray(search_arr.astype(np.uint8), mode="RGB").save(search_path_abs)
    else:
        Image.fromarray(ref_arr).save(ref_path_abs)
        Image.fromarray(search_arr).save(search_path_abs)

    return {
        "pair_id": pair_id, "architecture": architecture,
        "ref_path": ref_path_csv, "search_path": search_path_csv,
        "gt_x": round(gt_x, 2), "gt_y": round(gt_y, 2),
        "angle": round(angle, 2), "scale": DOWNSAMPLE,
        "add_marker": add_marker, "mode": mode
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", choices=["dram", "finfet", "both"], default="both")
    p.add_argument("--num_pairs", type=int, default=30)
    p.add_argument("--output_dir", default="../tests/synthetic_data")
    p.add_argument("--color", action="store_true", help="Generate RGB color images instead of grayscale (bonus feature)")
    p.add_argument("--jitter", action="store_true", help="Add ±10-20% random variation to shape sizes (bonus feature)")
    args = p.parse_args()

    archs = ["dram", "finfet"] if args.architecture == "both" else [args.architecture]
    rows = []
    arch_counts = {a: 0 for a in archs}
    
    mode = "RGB" if args.color else "L"
    for i in range(args.num_pairs):
        arch = archs[i % len(archs)]
        n = arch_counts[arch]
        add_marker = (n % 6 != 0)  # every 6th pair of THIS architecture is hard/periodic-only
        arch_counts[arch] += 1
        rows.append(generate_pair(arch, i, args.output_dir, add_marker=add_marker, mode=mode, jitter=args.jitter))
        jitter_str = ", jitter=True" if args.jitter else ""
        print(f"Generated pair {i} ({arch}, marker={add_marker}, mode={mode}{jitter_str})")

    csv_path = os.path.join(args.output_dir, "ground_truth.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    jitter_info = " with shape jitter" if args.jitter else ""
    print(f"\nSaved {len(rows)} pairs + ground_truth.csv to {args.output_dir} (mode={mode}{jitter_info})")

if __name__ == "__main__":
    main()