"""Hand-computed communication oracle, including asymmetric sender load."""
import tempfile
from pathlib import Path
import numpy as np
from measure import measure

with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'trace.bin'
    # 3 ranks, 2 samples/rank, pooling=2, ownership=id%3.
    rows=np.array([[0,0,1],[0,1,1],[0,0,3],[0,3,3],[0,0,1],[0,0,1]],dtype='<u4')
    rows.tofile(p)
    result,raw=measure([p],3,2,1,[2],0)
    assert raw['receive'].tolist()==[[[1,2,2]]]
    assert raw['send'].tolist()==[[[3,2,0]]]
    assert raw['hits'].tolist()==[[[1,0,0]]]
    assert result['local_hit_rate']==1/12
    assert result['unique_receive']['absolute_peak']==2
    assert result['unique_send']['absolute_peak']==3
    assert result['all_reduction_paper_per_rank']==6
    assert result['all_reduction_excluding_self_per_rank']==4
    try:
        measure([p],3,2,2,[2],0)
    except ValueError:
        pass
    else:
        raise AssertionError('short input accepted')
    paths=[]
    for r in range(3):
        q=Path(tmp)/f'dataset_{r}.bin';rows[r*2:r*2+2].tofile(q);paths.append(q)
    cps,craw=measure(paths,3,2,1,[2],0,True)
    assert np.array_equal(craw['receive'],raw['receive'])
    assert np.array_equal(craw['send'],raw['send'])
    # Four distinct remote FIDs per rank exceed the two remote Reduction vectors.
    # The same FID in another table must still be counted separately.
    reduction_rows=np.array([[0,1,3,1,3],[0,5,7,5,7],
                             [0,0,2,0,2],[0,4,6,4,6]],dtype='<u4')
    reduction_path=Path(tmp)/'reduction.bin';reduction_rows.tofile(reduction_path)
    reduction,counts=measure([reduction_path],2,2,1,[2,2],0)
    assert counts['unique'].tolist()==[[[4,4],[4,4]]]
    assert counts['receive'].tolist()==[[[4,4],[4,4]]]
    assert [x['strategy'] for x in reduction['per_table']]==['Reduction','Reduction']
    assert reduction['all_reduction_receive_per_rank']==4
    assert reduction['acp_receive']['mean']==4
print('PASS: exact traffic, table-scoped Unique, Reduction selection, locality, CPS parity, truncated batch rejection')
