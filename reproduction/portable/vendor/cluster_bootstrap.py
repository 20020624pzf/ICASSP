"""Paired registered-client bootstrap, conditional on fixed checkpoints."""
import numpy as np

def draw_macro(cluster_tables,replicates=10000,seed=1729,batch_size=25):
 rng=np.random.default_rng(seed);shape=cluster_tables[0][0].shape[1:];draw=np.zeros((replicates,*shape))
 for numerator,denominator in cluster_tables:
  assert numerator.ndim==3 and denominator.ndim==1 and len(numerator)==len(denominator)
  assert np.issubdtype(numerator.dtype,np.integer) and np.issubdtype(denominator.dtype,np.integer)
  for begin in range(0,replicates,batch_size):
   ids=rng.integers(0,len(denominator),size=(min(batch_size,replicates-begin),len(denominator)));D=denominator[ids].sum(1);assert np.all(D>0);N=numerator[ids].sum(1)
   draw[begin:begin+len(ids)]+=100*N/D[:,None,None]/len(cluster_tables)
 return draw
