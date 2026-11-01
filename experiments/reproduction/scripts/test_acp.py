"""Independent hand oracle and mismatched/truncated input rejection."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('acp',root/'working/acp/compute_acp.py');acp=importlib.util.module_from_spec(spec);spec.loader.exec_module(acp)
with tempfile.TemporaryDirectory() as tmp:
 d=Path(tmp);f=d/'input.bin'
 rows=np.array([[0,0,1],[0,1,1],[0,0,3],[0,3,3],[0,0,1],[0,0,1]],dtype='<i4');rows.tofile(f)
 remote,unique=acp.compute_unique_per_table_hash(f,3,2,1,[2],0)
 np.testing.assert_allclose(remote,[5/3]);np.testing.assert_allclose(unique,[2])
 for r in range(3):rows[r*2:r*2+2].tofile(d/f'dataset_{r}.bin')
 recv,uniq,peak,counts=acp.compute_unique_per_table_cps(d,3,2,1,[2],0,True)
 assert counts['receive'].tolist()==[[[1,2,2]]]
 assert counts['send'].tolist()==[[[3,2,0]]]
 assert peak.tolist()==[2]
 def reject(fn):
  try:fn()
  except (ValueError,FileNotFoundError):pass
  else:raise AssertionError('invalid input accepted')
 reject(lambda:acp.compute_unique_per_table_hash(f,3,2,2,[2],0))
 reject(lambda:acp.compute_unique_per_table_cps(d,3,2,2,[2],0))
 with f.open('ab') as out:out.write(b'x')
 reject(lambda:acp.compute_unique_per_table_hash(f,3,2,1,[2],0))
 (d/'dataset_1.bin').unlink()
 reject(lambda:acp.compute_unique_per_table_cps(d,3,2,1,[2],0))
print('PASS: zero-dense layout, exact send/receive, missing/truncated/unequal-range rejection')
