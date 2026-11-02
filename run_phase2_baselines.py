"""Top-level script to execute Phase 2 baseline evaluations and export results."""

import argparse
import sys
import os

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.experiments.baseline_runner import run_all_phase2_baselines


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 Baseline structural evaluations.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2_baselines.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    print(f"Executing Phase 2 baseline runs with config: {args.config}")
    df_all, df_summary = run_all_phase2_baselines(args.config)
    print("\n=== Phase 2 Baseline Results Summary Table ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

