"""Exact owner-change ledger for periodic incremental fine-tuning.
Logical IDs, not remapped physical addresses, identify embeddings. Tensor bytes
exclude routing/version metadata, which must be reported separately.
"""
import numpy as np

def migration_ledger(old_owner,new_owner,ranks,allowed,max_moved,dim=128,element_bytes=4,state_tensors=1):
 old=np.asarray(old_owner);new=np.asarray(new_owner)
 if old.shape!=new.shape:raise ValueError('Compare identical pre-existing logical ID sets')
 if np.any((old<0)|(old>=ranks)|(new<0)|(new>=ranks)):raise ValueError('Invalid owner')
 ids=np.flatnonzero(old!=new)
 if not allowed and len(ids):raise ValueError('Old embeddings moved outside a fine-tuning checkpoint')
 if len(ids)>max_moved:raise ValueError('Migration budget exceeded; both sides of a swap count')
 send=np.bincount(old[ids],minlength=ranks);receive=np.bincount(new[ids],minlength=ranks)
 pairs=np.bincount(old[ids]*ranks+new[ids],minlength=ranks*ranks).reshape(ranks,ranks)
 row_bytes=dim*element_bytes*state_tensors
 assert send.sum()==receive.sum()==len(ids) and np.trace(pairs)==0
 return dict(moved_embeddings=len(ids),logical_id_indices=ids.tolist(),send_rows=send.tolist(),receive_rows=receive.tolist(),routing_rows=pairs.tolist(),tensor_bytes_per_moved_embedding=row_bytes,total_network_tensor_bytes=int(len(ids)*row_bytes),peak_rank_send_tensor_bytes=int(send.max()*row_bytes),peak_rank_receive_tensor_bytes=int(receive.max()*row_bytes),storage_count_delta=(receive-send).tolist(),metadata_bytes_included=False,optimizer_state_tensors=state_tensors)

def checkpoint_due(completed_training_steps,last_checkpoint_step,min_interval):
 if min_interval<=0:raise ValueError('Interval must be positive')
 return completed_training_steps-last_checkpoint_step>=min_interval
