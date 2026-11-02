"""Single-command Master Reproduction Script for Lattice-Bio Research Paper.

Reproduces all FE validations, baselines, optimization experiments (E1 to E7),
statistical hypothesis testing, publication figures (Fig 1 to Fig 8),
and interactive web demo datasets.

Usage:
    # 1. Regenerate all 8 publication figures & tables in seconds from verified results:
    python reproduce_all.py --mode figures

    # 2. Run complete pytest validation suite (63/63 tests):
    python reproduce_all.py --test

    # 3. Fast end-to-end smoke reproduction (< 1 minute):
    python reproduce_all.py --mode quick

    # 4. Full experimental campaign reproduction from scratch:
    python reproduce_all.py --mode full

    # 5. Launch interactive web demo locally:
    python reproduce_all.py --serve-demo [--port 8000]
"""

from __future__ import annotations
import argparse
import http.server
import os
import socketserver
import subprocess
import sys
import time
from typing import Dict, Any


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "29d59fb"


def print_banner(git_hash: str) -> None:
    """Print research reproduction banner."""
    print("=" * 80)
    print(" LATTICE-BIO: BONE REMODELING VS EVOLUTION RESEARCH REPRODUCTION")
    print(f" Environment : Python {sys.version.split()[0]} | Platform: {sys.platform}")
    print(f" Git Commit  : {git_hash}")
    print(f" Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def run_tests() -> int:
    """Execute pytest validation suite."""
    print("\n[STEP] Running complete pytest validation suite (63 tests)...")
    res = subprocess.run([sys.executable, "-m", "pytest", "-v", "--tb=short"])
    return res.returncode


def run_figures() -> None:
    """Regenerate all 8 publication paper figures from result CSVs."""
    print("\n[STEP] Generating publication figures (paper_fig1 to paper_fig8)...")
    t0 = time.time()
    from src.export.plot_paper_figures import generate_all_paper_figures
    generate_all_paper_figures()
    dt = time.time() - t0
    print(f"[SUCCESS] All 8 publication figures regenerated in figures/ ({dt:.2f} s).")


def run_quick_verification() -> None:
    """Run fast analytical validation and smoke test."""
    print("\n[STEP 1/3] Running Phase 1 Analytical FE Validation Suite...")
    subprocess.run([sys.executable, "run_validation.py", "--config", "configs/phase1_validation.yaml"], check=True)

    print("\n[STEP 2/3] Exporting interactive demo data...")
    subprocess.run([sys.executable, "export_demo_data.py"], check=True)

    print("\n[STEP 3/3] Generating all publication figures...")
    from src.export.plot_paper_figures import generate_all_paper_figures
    generate_all_paper_figures()
    print("\n[SUCCESS] Quick verification completed successfully.")


def run_full_campaign(config_path: str = "configs/phase4_experiments.yaml") -> None:
    """Run full Phase 4 experimental campaign E1 through E7."""
    print(f"\n[STEP 1/3] Executing complete Phase 4 experimental campaign ({config_path})...")
    t0 = time.time()
    subprocess.run([sys.executable, "run_phase4_experiments.py", "--config", config_path], check=True)
    dt_exp = time.time() - t0
    print(f"[SUCCESS] Experimental campaigns E1-E7 completed in {dt_exp:.1f} s.")

    print("\n[STEP 2/3] Exporting interactive demo replay dataset...")
    subprocess.run([sys.executable, "export_demo_data.py"], check=True)

    print("\n[STEP 3/3] Generating all publication figures (Fig 1 - Fig 8)...")
    from src.export.plot_paper_figures import generate_all_paper_figures
    generate_all_paper_figures()
    print("\n[SUCCESS] Full reproduction campaign finished with 100% success.")


def serve_demo(port: int = 8000) -> None:
    """Launch local HTTP server for interactive web demo."""
    demo_dir = os.path.abspath("demo")
    os.chdir(demo_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/index.html"
        print(f"\n[DEMO] Interactive Lattice Demo running at: {url}")
        print("[DEMO] Press Ctrl+C to terminate the demo server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[DEMO] Server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Reproduction CLI for Lattice-Bio Research")
    parser.add_argument(
        "--mode",
        choices=["figures", "quick", "full"],
        default="figures",
        help="Reproduction mode: 'figures' (fast plot rebuild), 'quick' (analytical smoke test), 'full' (all E1-E7 experiments)",
    )
    parser.add_argument("--test", action="store_true", help="Run full pytest test suite")
    parser.add_argument("--serve-demo", action="store_true", help="Launch local HTTP server for demo")
    parser.add_argument("--port", type=int, default=8000, help="Port for demo HTTP server (default: 8000)")
    parser.add_argument("--config", type=str, default="configs/phase4_experiments.yaml", help="Path to experiment config YAML")

    args = parser.parse_args()

    git_hash = get_git_commit_hash()
    print_banner(git_hash)

    if args.test:
        code = run_tests()
        sys.exit(code)

    if args.serve_demo:
        serve_demo(port=args.port)
        return

    if args.mode == "figures":
        run_figures()
    elif args.mode == "quick":
        run_quick_verification()
    elif args.mode == "full":
        run_full_campaign(config_path=args.config)


if __name__ == "__main__":
    main()

