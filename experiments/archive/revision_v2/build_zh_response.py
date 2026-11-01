"""Rebuild the current V3 response and update this legacy entry point."""
from pathlib import Path
import subprocess,sys,shutil
V=Path(__file__).resolve().parents[2]/'zh_revision_v3'
for script in ['build_response.py','validate_deliverables.py','publish_local_docs.py']:
 subprocess.run([sys.executable,str(V/'code'/script)],check=True)
