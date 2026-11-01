from pathlib import Path
import csv,json
V=Path(__file__).resolve().parents[1]
methods=['HugeCTR','TorchRec-auto','EMBark','HET-GMP','Herald','BagPipe','RAPID'];core=['HugeCTR','TorchRec-auto','EMBark','RAPID'];real=[('Kaggle','DLRM-DCNv2'),('Terabyte','DLRM-DCNv2'),('Taobao','DLRM-DCNv2'),('MovieLens-25M','DLRM')];secondary=[(d,'DLRM') for d in ['Kaggle','Terabyte','Taobao']];synth=[(d,'DLRM-DCNv2-inspired') for d in ['T180-inspired','T200-inspired','T510-inspired']]
fields=['experiment','dataset','model','method','repeat','seed','gpus','local_batch','accumulation','precision','epochs','learning_rate','samples_per_second','step_ms_mean','step_ms_p95','peak_gpu_memory_bytes','preprocessing_seconds','total_training_seconds','auc','logloss','auc_paired_difference','embedding_communication_bytes','rank_mean_bytes','rank_peak_bytes','peak_mean_ratio','status','manifest_path','log_path','trace_path','notes'];rows=[]
def add(kind,ds,model,method,repeat='',seed='',batch=8192,acc=1):
 row={k:'' for k in fields};row.update(experiment=kind,dataset=ds,model=model,method=method,repeat=repeat,seed=seed,gpus=8,local_batch=batch,accumulation=acc,precision='FP32',status='待运行');rows.append(row);return row
for ds,model in real+synth:
 for m in methods:
  for repeat in [1,2,3]:
   r=add('主性能',ds,model,m,repeat);r['notes']='合成配置须先核对每表manifest，不宣称原版EMBark复现' if ds.endswith('inspired') else '运行前完成模型适配与一致性检查'
for ds,model in secondary:
 for m in core:
  for repeat in [1,2,3]:add('第二模型性能',ds,model,m,repeat)
for ds,model in real+secondary:
 for m in ['HugeCTR','RAPID-CPS','RAPID-ACP','RAPID-CPS+ACP']:
  for seed in [7,19,41,73,101]:
   r=add('完整精度',ds,model,m,seed=seed);r['epochs']=5;r['notes']='至少5个完整epoch；若reference未收敛，所有配对方法延长至同一epoch'
for ds,model in real:
 for m in ['HugeCTR','RAPID']:
  for batch in [512,1024,2048,4096,8192,16384]:
   for repeat in [1,2,3]:add('batch敏感性',ds,model,m,repeat,batch=batch)
  for batch,acc in [(1024,8),(2048,4),(4096,2),(8192,1)]:
   for repeat in [1,2,3]:add('固定有效batch梯度累积',ds,model,m,repeat,batch=batch,acc=acc)
with open(V/'训练结果待填.csv','w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(V/'experiment_plan.json').write_text(json.dumps(dict(methods=methods,real_main=real,real_secondary=secondary,synthetic=synth,accuracy_seeds=[7,19,41,73,101],performance_repeats=3,performance_warmup_steps=100,performance_measured_steps_min=1000,accuracy_epochs_min=5,auc_equivalence_margin=0.001,incremental_operations=['admit_new_embeddings','schedule_samples_within_original_global_batch'],old_embedding_migration=False,physical_multinode_claim=False,training_matrix_rows=len(rows),measured_gpu_results_present=False),ensure_ascii=False,indent=2))
print(len(rows))
