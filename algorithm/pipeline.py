"""
Drift-Sense end-to-end pipeline: generate -> infer -> evaluate -> report.
Usage:
    python pipeline.py --config config.yaml
    python pipeline.py --config config.yaml --skip-generate
"""
import argparse
import logging
import subprocess
import sys
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("drift-sense")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_step(name, cmd, cwd=None):
    log.info(f"--- Starting: {name} ---")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        log.error(f"{name} FAILED (exit code {result.returncode})")
        sys.exit(1)
    log.info(f"--- Completed: {name} ---")


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense full pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info(f"Loaded config from {args.config}")

    if not args.skip_generate:
        ds = cfg["dataset"]
        gen_cmd = [
            "python", "dataset/generate_dataset.py",
            "--architecture", "both",
            "--num_pairs", str(ds["num_pairs"]),
            "--output_dir", ds["output_dir"],
        ]
        if ds.get("jitter"):
            gen_cmd.append("--jitter")
        run_step("Dataset generation", gen_cmd)
    else:
        log.info("Skipping dataset generation (--skip-generate)")

    run_step("Batch evaluation", ["python", "batch_eval.py"], cwd="core")
    run_step("Accuracy plot", ["python", "plot_pr_curve.py"], cwd="core")

    log.info("Pipeline complete. See core/batch_results.csv and core/pr_curve.png")


if __name__ == "__main__":
    main()