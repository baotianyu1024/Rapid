"""Compatibility entrypoint: execute the current offline RAPID baseline."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
os.execv(sys.executable,[sys.executable,str(root/'scripts/run_offline.py'),*sys.argv[1:]])
