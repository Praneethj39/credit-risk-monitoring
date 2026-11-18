"""One command for the whole pipeline: data -> features -> train -> monitor."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"

for step in ["train.py", "monitor.py"]:
    print(f"\n===== {step} =====")
    subprocess.run([sys.executable, step], cwd=SRC, check=True)
