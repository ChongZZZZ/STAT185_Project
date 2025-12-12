# run_all.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
python = sys.executable  # this will be your conda env's python
subprocess.run([python, str(ROOT / "mnist_anomaly.py")], check=True)
subprocess.run([python, str(ROOT / "manufacture.py")], check=True)

