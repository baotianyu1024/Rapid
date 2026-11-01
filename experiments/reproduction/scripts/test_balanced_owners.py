"""Check migration predictions against independent requester/owner enumeration."""
import unittest
import tempfile
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from refine_balanced_owners import deltas, materialize, best_swap
from measure import measure


class MigrationTest(unittest.TestCase):
    def test_swap_preserves_receiver_and_storage(self):
        # Both IDs occur on ranks 0 and 1, but only ID 0 occurs on ranks 2,3.
        profiles=np.array([[1,1,1,1],[1,1,0,0]],dtype=np.uint16)
        weights=np.array([[4],[2]],dtype=np.uint16)
        owners=np.array([0,1],dtype=np.int16)
        tables=[dict(profiles=profiles,weights=weights,owners=owners,mass=weights.sum(1))]
        send=np.array([[3,1,0,0]])
        result=best_swap(tables,send[:,None,:],send,np.array([3.0]),1)
        self.assertIsNone(result)  # Already within the configured cap.
        result=best_swap(tables,send[:,None,:],send,np.array([2.0]),1)
        # A whole-ID swap would overload the other owner, so reject it.
        self.assertIsNone(result)
        tables=[dict(profiles=np.array([[1,1,1,1],[1,1,1,1],[1,1,0,0]],dtype=np.uint16),
                     weights=np.array([[4],[4],[2]],dtype=np.uint16),
                     owners=np.array([0,0,1]),mass=np.array([4,4,2]))]
        send=np.array([[6,1,0,0]])
        t,s,d,hi,lo,delta=best_swap(tables,send[:,None,:],send,np.array([4.0]),1)
        np.testing.assert_array_equal(delta,[2])
        self.assertEqual((s,d,len(hi),len(lo)),(0,1,1,1))
        for rank in (s,d):
            np.testing.assert_array_equal(tables[t]['profiles'][hi,rank],tables[t]['profiles'][lo,rank])

    def test_materialized_records_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'base';out=root/'out'
            (case/'scheduled').mkdir(parents=True);out.mkdir()
            cfg=dict(ranks=3,batches=2,batch_size=4,dense=1,table_lengths=[2])
            (case/'config.json').write_text(json.dumps(cfg))
            rng=np.random.default_rng(7)
            ids=np.array([0,2,4,7,9],dtype=np.uint32)
            owners=np.array([2,1,0,2,0])
            receive=np.zeros((2,1,3),dtype=np.int64);send=np.zeros_like(receive)
            for r in range(3):
                rows=np.column_stack((np.arange(8)+r*8,rng.integers(0,100,8),rng.choice(ids,(8,2)))).astype('<u4')
                rows.tofile(case/'scheduled'/f'dataset_{r}.bin')
                for b in range(2):
                    for uid in set(rows[b*4:(b+1)*4,2:].ravel()):
                        owner=int(owners[list(ids).index(uid)])
                        if owner!=r:
                            receive[b,0,r]+=1;send[b,0,owner]+=1
            _,counts=measure([case/'scheduled'/f'dataset_{r}.bin' for r in range(3)],3,4,2,[2],1,True)
            np.savez(case/'cps_counts.npz',**counts)
            materialize(case,root,out,cfg,[dict(ids=ids,owners=owners)],receive,send)
            self.assertTrue((out/'complete.json').exists())
            timing=json.loads((out/'materialization.json').read_text())
            self.assertGreater(timing['seconds'],0);self.assertLess(timing['seconds'],60)
            plain=root/'plain';plain.mkdir()
            materialize(case,root,plain,cfg,[dict(ids=ids,owners=owners)],receive,send,compressed=False)
            for r in range(3):
                self.assertEqual((out/'scheduled'/f'dataset_{r}.bin').read_bytes(),(plain/'scheduled'/f'dataset_{r}.bin').read_bytes())
            with np.load(out/'table_0_remap.npz') as packed,np.load(plain/'table_0_remap.npz') as stored:
                for key in packed.files:np.testing.assert_array_equal(packed[key],stored[key])

            optimized=root/'optimized'
            subprocess.run([sys.executable,str(Path(__file__).with_name('refine_balanced_owners.py')),
                            '--case',str(case),'--cache',str(root/'cache'),'--output',str(optimized),
                            '--rounds','2','--materialize'],check=True,stdout=subprocess.DEVNULL)
            self.assertTrue(json.loads((optimized/'complete.json').read_text())['all_records_inverse_verified'])

    def test_random_presence_and_locality_changes(self):
        rng = np.random.default_rng(123)
        for _ in range(100):
            n, nb, g = 19, 5, 4
            presence = rng.random((n,nb,g)) < .35
            owner = rng.integers(0,g,n)
            s,d = rng.choice(g,2,replace=False)
            ix = np.flatnonzero(owner==s)
            powers = np.uint64(1)<<np.arange(nb,dtype=np.uint64)
            profile = (presence*powers[None,:,None]).sum(axis=1,dtype=np.uint64)
            ls,ld,ds,dd = deltas(profile[ix],presence[ix].sum(axis=2),s,d,nb)

            def enumerate_counts(owners):
                receive=np.zeros((nb,g),dtype=np.int64)
                send=np.zeros_like(receive)
                for i in range(n):
                    for b in range(nb):
                        for r in range(g):
                            if presence[i,b,r] and r!=owners[i]:
                                receive[b,r]+=1
                                send[b,owners[i]]+=1
                return receive,send

            receive,send=enumerate_counts(owner)
            receive[:,s]+=ls.sum(0);receive[:,d]-=ld.sum(0)
            send[:,s]-=ds.sum(0);send[:,d]+=dd.sum(0)
            owner[ix]=d
            expected_r,expected_s=enumerate_counts(owner)
            np.testing.assert_array_equal(receive,expected_r)
            np.testing.assert_array_equal(send,expected_s)


if __name__=='__main__':
    unittest.main()
