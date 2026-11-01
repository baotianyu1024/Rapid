"""Combine all comparison figures and the complete numeric ledger in a single PDF."""
import datetime,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,'/tmp/rapid_pdf_deps')
from pypdf import PdfWriter,PdfReader
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figure_comparison'
rows=json.loads((OUT/'all_numeric_comparisons.json').read_text())
scopes=sorted(set(r['scope'] for r in rows));ids={s:f'C{i+1}' for i,s in enumerate(scopes)}
ledger=OUT/'numeric_ledger.pdf'
with PdfPages(ledger) as pdf:
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.91,'RAPID: paper versus real-data reproduction',fontsize=20)
 text='\n'.join([
  'Generated '+datetime.datetime.now().isoformat(timespec='seconds'),
  '', 'Scope: CPU replay of actual initialized and shuffled datasets; no V100 training.',
  'Criteo: complete 2015-02-21, 204,846,845 rows; not all 24 days.',
  'Kaggle: complete training data, 45,840,617 rows.',
  'Taobao: Display Advertising dataset, 26,557,961 rows (user-approved code choice).',
  'Main Criteo/Kaggle scaling traces: 4,194,304 rows per rank configuration.',
  'Taobao includes all complete 8-rank batches plus 20-batch scaling cases.',
  '', 'Excluded: Figures2/8/9 runtime/throughput; Figure12(a) AUC requires training.',
  'Figures1/3/5/7 and Figure4(a) are workflow/architecture diagrams, not numeric experiments.',
  'Covered: Figures4(b,c),6(a,b,c),10,11,12(b), Tables2/3 and Section7.8 load balance.',
  '', 'Lower traffic and higher locality are favorable; better values are not tuned down.',
  'Dataset-distribution statistics are descriptive, not algorithm pass/fail targets.',
  'MCR is estimated from sampled cold IDs in a one-million-row trace, with scope recorded.',
  'Raw-access and unique-ID locality denominators are both shown; never interchanged.',
  'Receive and send balance are separate; receive balance does not prove send balance.',
  'Figure4 Reduction conflicts with Eq.(5)/Figure11 dense model.',
  'Traffic counts remote receive only: Reduction B*(G-1), self rank excluded.',
  'Figure12 labels total unique IDs as remote; total and actual remote counts are retained.',
  'Figure10 ACP/RAPID is effective traffic reduction when Reduction tables are selected.',
  '', 'Reference curves: supplied plotting constants; other values: printed paper labels.',
  'Measurements never read reference values. Exact figures/units/scopes are in the CSV ledger.',
  'Complete source/config/logs: reproduction/; Chinese index: figure_comparison/INDEX.txt.',
  'Decision summary: FINAL_FINDINGS.txt and final_selected_results.csv.',
  'Selected candidates satisfy receive ratio <=1.20 where possible, then minimize receive peak.'
 ])
 fig.text(.06,.85,text,fontsize=10,va='top',linespacing=1.6);pdf.savefig(fig);plt.close(fig)
 for start in range(0,len(scopes),22):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.05,.94,'Experiment scope index',fontsize=16)
  fig.text(.05,.89,'\n\n'.join(f'{ids[s]}: {s}' for s in scopes[start:start+22]),fontsize=8,va='top',family='monospace');pdf.savefig(fig);plt.close(fig)
 for start in range(0,len(rows),27):
  page=rows[start:start+27];cells=[];colors=[]
  for r in page:
   paper='n/a' if r['paper'] is None else f'{r["paper"]:.6g}';meas=f'{r["measured"]:.7g}'
   state='descriptive'
   if r['paper'] is None:state='measured only'
   elif r['direction']!='descriptive':
    tol=r.get('reference_rounding_tolerance',0);ok=r['measured']<=r['paper']+tol if r['direction']=='lower' else r['measured']>=r['paper']-tol;state='better/equal' if ok else 'below target'
   cells.append([r['figure']+'/'+r['panel'],r['metric'],paper,meas,state,ids[r['scope']]])
   colors.append(['#ffeded' if state=='below target' else '#edf8ed' if state=='better/equal' else 'white']*6)
  fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(f'Complete numerical comparison: rows {start+1}-{start+len(page)} / {len(rows)}',loc='left',fontsize=14)
  t=ax.table(cellText=cells,colLabels=['Figure/panel','Metric','Paper','Measured','Assessment','Scope'],cellColours=colors,colWidths=[.13,.37,.12,.13,.15,.07],loc='center');t.auto_set_font_size(False);t.set_fontsize(6.5);t.scale(1,1.5)
  fig.tight_layout();pdf.savefig(fig);plt.close(fig)
writer=PdfWriter();writer.append(ledger,pages=(0,1))
for name in ['MainResults','Figure4','Figure6','Figure10','Figure11','Figure11Deviation','Figure12b','Table3','LoadBalance','StorageBalance']:
 p=OUT/f'{name}.pdf'
 if p.exists():writer.append(p)
writer.append(ledger,pages=(1,len(PdfReader(ledger).pages)))
with (ROOT/'paper_vs_measured_report.pdf').open('wb') as f:writer.write(f)
print('PDF report:',len(writer.pages),'pages,',len(rows),'numeric comparisons')
