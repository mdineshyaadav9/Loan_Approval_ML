# ═══════════════════════════════════════════════════════════════
# run_all.py — Run All Phases in One Go
# ═══════════════════════════════════════════════════════════════
#
# This script runs Phase 2 → 3 → 4 → 5 in sequence,
# then tells you how to start the frontend.
#
# USE: python run_all.py
# ═══════════════════════════════════════════════════════════════

import subprocess
import sys
import os

def run_phase(script, name):
    print(f"\n{'='*60}")
    print(f"  Running {name}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ ERROR in {name}. Fix the error above and try again.")
        sys.exit(1)
    print(f"\n✓ {name} completed successfully!")

if __name__ == "__main__":
    # Ensure we're in the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("🚀 LoanIQ ML Pipeline — Running All Phases")
    print("This will take 3-5 minutes for hyperparameter tuning.\n")

    run_phase("phase2_eda_cleaning.py",  "Phase 2 — EDA & Data Cleaning")
    run_phase("phase3_preprocessing.py", "Phase 3 — Preprocessing")
    run_phase("phase4_train_models.py",  "Phase 4 — Train Models")
    run_phase("phase5_boost_score.py",   "Phase 5 — Optimise")

    print("\n" + "="*60)
    print("✅ ALL PHASES COMPLETE! Your ML model is ready.")
    print("="*60)
    print("\nTo launch the frontend app, run:")
    print("  streamlit run frontend/app.py")
    print("\nThe app will open in your browser at http://localhost:8501")
