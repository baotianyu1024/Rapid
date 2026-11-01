import argparse,os,shutil,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();assert not a.out.exists();metrics=json.load(open(a.source/'metrics.json'));last=metrics[-1]['window'];assert (a.source/'checkpoint.pkl').is_file() and (a.source/f'window_{last:02d}/profile_cache').is_dir()
def copy(src,dst):
 if Path(src).suffix=='.json':return shutil.copy2(src,dst)
 os.link(src,dst);return dst
shutil.copytree(a.source,a.out,copy_function=copy);(a.out/'branch.json').write_text(json.dumps(dict(shared_prefix=str(a.source.resolve()),completed_windows=last,shared_files='immutable array/checkpoint hardlinks; JSON files copied; later checkpoints atomically replaced'),indent=2));print('BRANCHED',last,a.out)
