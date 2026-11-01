"""Export reviewable source changes against the user's preserved update_Rapid copy."""
from pathlib import Path
import difflib,json,hashlib
R=Path(__file__).resolve().parents[1];base=R.parent/'code/update_Rapid';work=R/'working';parts=[];manifest=[]
paths=sorted({p.relative_to(root) for root in [base,work] for p in root.rglob('*') if p.is_file() and p.suffix in ['.cpp','.h','.py','.sh'] and '.git' not in p.parts and '__pycache__' not in p.parts})
for rel in paths:
 old=base/rel;new=work/rel;a=old.read_text() if old.exists() else '';b=new.read_text() if new.exists() else ''
 if a==b:continue
 parts.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+str(rel),tofile='b/'+str(rel)))
 manifest.append(dict(path=str(rel),baseline_sha256=hashlib.sha256(a.encode()).hexdigest() if old.exists() else None,working_sha256=hashlib.sha256(b.encode()).hexdigest() if new.exists() else None))
(R/'working_changes.patch').write_text(''.join(parts));(R/'working_changes_manifest.json').write_text(json.dumps(manifest,indent=2));print('Changed source files:',len(manifest))
