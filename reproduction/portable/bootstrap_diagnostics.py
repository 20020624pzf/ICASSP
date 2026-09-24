"""Reproduce the four already reported CV27 diagnostics from anonymous counts."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
# Resolve only our loader by file path; never prepend generic modules to sys.path.
import importlib.util as _import_util
_loader_spec = _import_util.spec_from_file_location("_asr_vendor_loader", HERE / "asr_vendor_loader.py")
_loader = _import_util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(_loader)
load_vendor = _loader.load_vendor
draw_macro = load_vendor("cluster_bootstrap").draw_macro

def run(counts,out):
    data=np.load(counts,allow_pickle=False)
    expected=['Conf','Global','Text','Text+Global','ConfDupClean','ConfPerm']
    assert data['methods'].tolist()==expected
    tables=[];point=np.zeros((3,6))
    for l in ['ar','fa','pl','tr','it']:
        x=data[l]
        assert x.ndim==4 and x.shape[1:]==(3,6,4) and np.issubdtype(x.dtype,np.integer)
        den=x[:,:,0,3]
        assert np.array_equal(x[:,:,:,3],np.broadcast_to(den[:,:,None],x[:,:,:,3].shape))
        assert np.array_equal(den[:,0],den[:,1]) and np.array_equal(den[:,0],den[:,2])
        num=x[:,:,:,:3].sum(3)
        point+=100*num.sum(0)/den.sum(0)[:,None]/5
        tables.append((num,den[:,0]))
    draws=draw_macro(tables,10000,1729).mean(1);point=point.mean(0)
    vectors={'D':[1,-1,-1,1,0,0],'Global-ConfDupClean':[0,1,0,0,-1,0],
             'ConfPerm-Conf':[-1,0,0,0,0,1],'Global-ConfPerm':[0,1,0,0,0,-1]}
    result=[]
    for name,v in vectors.items():
        dd=draws@np.asarray(v)
        result.append(dict(name=name,estimate=float(point@v),CI95=np.quantile(dd,[.025,.975]).tolist(),
                           CI98_75=np.quantile(dd,[.00625,.99375]).tolist()))
    with Path(out).open('x') as f:
        json.dump({'contrasts':result,'family_size':4,'replicates':10000,'seed':1729,
                   'scope':'Post-exposure; fixed fits/checkpoints, not development-wide correction'},f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',default=str(HERE.parent/'data/CV27_CLUSTER_COUNTS.npz'))
    p.add_argument('--out',required=True)
    a=p.parse_args();run(a.counts,a.out)
