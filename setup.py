import os
import subprocess
import sys

def setup():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models', 'best_model.pkl'
    )

    if os.path.exists(model_path):
        print("Model already trained — skipping setup")
        return

    print("First run — training model (3-5 minutes)...")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    phases = [
        'phase2_eda_cleaning.py',
        'phase3_preprocessing.py',
        'phase4_train_models.py',
        'phase5_boost_score.py',
    ]

    for phase in phases:
        phase_path = os.path.join(base_dir, phase)
        print(f"Running {phase}...")
        result = subprocess.run(
            [sys.executable, phase_path],
            capture_output=True,
            text=True,
            cwd=base_dir
        )
        if result.returncode != 0:
            print(f"Error in {phase}:")
            print(result.stderr)
            sys.exit(1)
        print(f"Done: {phase}")

    print("Model training complete!")

if __name__ == "__main__":
    setup()
