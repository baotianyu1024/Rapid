from pathlib import Path
import json,csv,hashlib,re
V=Path(__file__).resolve().parents[1];H=V.parent;read=lambda p:json.loads(p.read_text());E=V/'results/incremental_extension';T=read(E/'summary.json');td={(r['policy'],r['window']):r for r in T['results']}
fields=['window','pure_unique','frozen','incremental','full_rebuild','incremental_over_full','incremental_saving_vs_unique','incremental_old_owner_changes','incremental_old_physical_changes','new_embeddings'];rows=[]
for k in range(1,T['windows']+1):
 row=dict(window=k)
 for p in ['pure_unique','frozen','incremental','full_rebuild']:row[p]=td[p,k]['metrics']['vectors_per_8192'] if (p,k) in td else ''
 row['incremental_over_full']=row['incremental']/row['full_rebuild'] if row['full_rebuild'] else '';row['incremental_saving_vs_unique']=1-row['incremental']/row['pure_unique'];r=td['incremental',k];row.update(incremental_old_owner_changes=r['old_owner_changes'],incremental_old_physical_changes=r['old_physical_changes'],new_embeddings=r['new_embeddings']);rows.append(row)
with open(V/'逐窗口通信.csv','w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
paths=[V/'results/mcr_support/result.json',V/'results/batch_unique/comparison.json',E/'summary.json',E/'stability.json',V/'results/wide_profiles_validation.json',H/'results/sensitivity/summary.json']+[H/f'results/{c}/comparison.json' for c in ['criteo_8','criteo_64','kaggle_8','taobao_8']]
manifest=[]
for p in paths+sorted((V/'code').glob('*')):
 if p.is_file():manifest.append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(V/'evidence_manifest.json').write_text(json.dumps(dict(artifacts=manifest,physical_training_results=False,old_embedding_migration=False,incremental_unit='F+B vectors per8192samples; each network payload counted once; excludes local/index/metadata/denseAllReduce',new_window_rows=65536,profile_rows=T['profile_rows'],windows=T['windows'],full_rebuild_only_at=T['full_checkpoints'],source_time_order=False),ensure_ascii=False,indent=2))
md=(V/'中文Response.md').read_text();assert not any(ord(c)<32 and c not in '\n\t' for c in md);assert '【运行中】' not in md;assert len(read(V/'review_coverage.json'))==37
for label,url in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',md):
 if not url.startswith(('https://','http://','#')):assert (V/url).exists(),url
assert all(r['incremental_old_owner_changes']==r['incremental_old_physical_changes']==0 for r in rows)
assert read(V/'results/batch_unique/complete.json')['success'] and read(E/'complete.json')['success']
print('PASS: complete local experiments, 37 review entries, links, math control characters, zero old embedding migration; GPU metrics intentionally pending')
