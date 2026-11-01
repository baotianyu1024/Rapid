"""Update the established Markdown paths after validation; preserve previous text."""
from pathlib import Path
import shutil,json
V=Path(__file__).resolve().parents[1];old=V.parent/'zh_revision_v2';assert json.loads((V/'validation.json').read_text())['success'];backup=old/'archive_before_20261027';backup.mkdir(exist_ok=True)
for name in ['中文Response.md','训练实验方案.md','训练结果待填.csv']:
 if not (backup/name).exists():shutil.copy2(old/name,backup/name)
 # Keep relative reviewer-facing links working in the established directory.
 
 if (old/name).is_symlink() or (old/name).exists():(old/name).unlink()
 (old/name).symlink_to(Path('../zh_revision_v3')/name)
for name in ['实验口径与复现说明.md','review_coverage.json','通信量对比.csv','BagPipe窗口与通信分解.csv','官方维度通信量对比.csv']:
 if (old/name).exists() and not (backup/name).exists():shutil.copy2(old/name,backup/name)
 if (old/name).is_symlink() or (old/name).exists():(old/name).unlink()
 (old/name).symlink_to(Path('../zh_revision_v3')/name)
p=old/'code/build_zh_response.py'
if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
p.write_text('''"""Rebuild the current V3 response and update this legacy entry point."""
from pathlib import Path
import subprocess,sys,shutil
V=Path(__file__).resolve().parents[2]/'zh_revision_v3'
for script in ['build_response.py','validate_deliverables.py','publish_local_docs.py']:
 subprocess.run([sys.executable,str(V/'code'/script)],check=True)
''')
(old/'CURRENT_REVISION.md').write_text('当前中文回复已同步为 zh_revision_v3。新通信实验、完整 RAPID 参数实验与全量 MCR 的证据位于 ../zh_revision_v3/results；原始 v2 结果保留作追溯，不再作为当前参数／通信表的来源。\n')
print('Updated',old/'中文Response.md')
