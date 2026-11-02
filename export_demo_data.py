"""Generate precomputed JSON replay dataset for the static web demo."""

import os
import sys
import time

from src.export.json_exporter import export_demo_dataset

def main():
    t0 = time.time()
    out_path = "demo/data/lattice_data.json"
    print("=" * 80)
    print(" Exporting Precomputed Optimization Trajectories for Web Demo")
    print(f" Target Output: {out_path}")
    print("=" * 80)

    data = export_demo_dataset(
        config_path="configs/phase4_experiments.yaml",
        out_path=out_path,
        seed=42,
    )
    print(f"\nDemo data export completed in {time.time() - t0:.2f} s")
    print(f"Total scenarios exported: {len(data['scenarios'])}")

if __name__ == "__main__":
    main()

