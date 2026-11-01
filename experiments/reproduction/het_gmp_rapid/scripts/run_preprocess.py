import argparse
from common import *
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--algorithm',choices=['rapid','het'],required=True); a=p.parse_args()
data=Path(a.data).resolve(); out=Path(a.out).resolve(); cfg=json.loads((data/'input.json').read_text()); nb=cfg['global_batches']
if a.algorithm=='het':
    measured('partition',[sys.executable,ROOT/'scripts/partition_het.py','--data',data,'--out',out],out)
else:
    src=ROOT/'src/rapid'; args=['--table_lengths',*LENGTHS,'--dense_count',DENSE]
    measured('hot',[sys.executable,src/'cps/hot_id_extraction.py','--input',data/'input.bin','--output',out/'hot','--topk',40000,'--max_rows',min(nb,80)*P*B,*args],out)
    measured('placement',[src/'cps/arrange_ids_to_devices','--input_bin',data/'input.bin','--hot_ids_dir',out/'hot','--output_dir',out/'placement','--batch_size',B,'--num_devices',P,'--n_batches',nb,*args],out)
    measured('schedule',[src/'cps/generate_local_dataset','--input_file',data/'input.bin','--table_files_dir',out/'placement','--output_dir',out/'scheduled','--num_devices',P,'--train_batch_size',B,'--steps_per_chunk',1,'--num_chunks',nb,'--num_threads',8,'--gamma',.2,*args],out,threads=8)
    measured('acp',[sys.executable,src/'acp/compute_acp.py','--input',data/'input.bin','--num_gpus',P,'--batch_size',B,'--num_batches',50,'--cps_dataset_dirs',out/'scheduled','--output',out/'acp.json',*args],out)
