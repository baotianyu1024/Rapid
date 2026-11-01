"""Compatibility entry: official HugeCTR generator only."""
import runpy,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'official_tseries/code/generate.py'
runpy.run_path(str(p),run_name='__main__')
