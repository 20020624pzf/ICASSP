"""Small fixed-bank expected-edit-risk selector, CPU/numpy/scipy only.

This is a tested mathematical prototype, NOT an ASR experiment or a new algorithm.
Use original cached feature coordinates/scales. This module does not read audio,
create candidates, open TEST references, or verify provenance on behalf of callers.
References/error labels are accepted by fit(), never by choose().
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import minimize

@dataclass(frozen=True)
class FitBank:
    language: str
    features: np.ndarray    # unstandardized (candidate, dimension); beam is row 0
    edits: np.ndarray       # full word S+D+I, including beam
    reference_words: int

@dataclass(frozen=True)
class Model:
    scale: np.ndarray
    coef: np.ndarray
    def choose(self, features: np.ndarray) -> int:
        x = _features(features, len(self.coef))
        v = (x / self.scale) @ self.coef
        if not np.isfinite(v).all():
            raise ValueError('nonfinite inference scores')
        j = int(np.argmax(v))
        return j if v[j] > 0 else 0

def _features(x, dim):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not 1 <= len(x) <= 6 or x.shape[1] != dim:
        raise ValueError('invalid fixed-bank feature shape')
    if not np.isfinite(x).all() or not np.array_equal(x[0], np.zeros(dim)):
        raise ValueError('features must be finite; beam row exactly zero')
    return x

def prepare(banks: Sequence[FitBank], scale):
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim != 1 or len(scale)==0 or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('invalid inherited feature scale')
    if not banks:
        raise ValueError('empty FIT set')
    denom = {}
    xlist=[]; elist=[]
    for b in banks:
        if not isinstance(b.language,str) or not b.language:
            raise ValueError('missing language')
        if isinstance(b.reference_words,bool) or not isinstance(b.reference_words,(int,np.integer)) or b.reference_words<0:
            raise ValueError('invalid reference word count')
        x = _features(b.features,len(scale))
        e = np.asarray(b.edits,dtype=np.float64)
        if e.shape != (len(x),) or not np.isfinite(e).all() or np.any(e<0) or not np.array_equal(e,np.floor(e)):
            raise ValueError('word edits must be finite nonnegative integers')
        denom[b.language] = denom.get(b.language,0)+int(b.reference_words)
        xlist.append(x/scale);elist.append(e-e[0])
    if any(n<=0 for n in denom.values()):
        raise ValueError('zero language-level FIT reference denominator')
    weights=np.array([1/denom[b.language] for b in banks],dtype=np.float64)
    weights/=weights.sum()  # equivalent to fixed macro edit risk, up to one constant
    k=max(map(len,xlist)); n=len(banks); d=len(scale)
    x=np.zeros((n,k,d)); e=np.zeros((n,k)); mask=np.zeros((n,k),dtype=bool)
    for i,(xx,ee) in enumerate(zip(xlist,elist)):
        x[i,:len(xx)]=xx; e[i,:len(ee)]=ee; mask[i,:len(xx)]=True
    return x,e,mask,weights,denom

def loss_grad(theta, prepared, penalty=0.01):
    x,e,mask,a,_=prepared
    theta=np.asarray(theta,dtype=np.float64)
    if theta.shape!=(x.shape[2],) or not np.isfinite(theta).all() or not np.isfinite(penalty) or penalty<=0:
        raise ValueError('invalid parameters or penalty')
    scores=np.einsum('nkd,d->nk',x,theta)
    if not np.isfinite(scores).all(): raise FloatingPointError('nonfinite logits')
    scores=np.where(mask,scores,-np.inf)
    mx=scores.max(axis=1,keepdims=True)
    p=np.exp(scores-mx);p/=p.sum(axis=1,keepdims=True)
    expected=np.sum(p*e,axis=1)
    value=float(a@expected+penalty*(theta@theta))
    coefficient=a[:,None]*p*(e-expected[:,None])
    grad=np.einsum('nk,nkd->d',coefficient,x)+2*penalty*theta
    return value,grad

def fit(banks: Sequence[FitBank], scale, inherited_ridge_coef, *, source_split: str,
        penalty: float=0.01, maxiter: int=2000):
    if source_split != 'TRAIN_CAL_FIT':
        raise ValueError('FIT only; caller must verify actual data provenance')
    prep=prepare(banks,scale)
    w0=np.asarray(inherited_ridge_coef,dtype=np.float64)
    if w0.shape!=(len(scale),) or not np.isfinite(w0).all():
        raise ValueError('invalid inherited ridge coefficients')
    histories=[]; options=[]
    # Both starts are predeclared; selection uses FIT penalized risk only.
    for name,start in [('zero',np.zeros_like(w0)),('inherited_ridge',w0.copy())]:
        trajectory=[]
        def trace(w):
            v,g=loss_grad(w,prep,penalty)
            trajectory.append({'iteration':len(trajectory),'objective':v,'gradient_inf':float(np.max(np.abs(g))),'coef':w.tolist()})
        trace(start)
        r=minimize(lambda w:loss_grad(w,prep,penalty),start,jac=True,method='L-BFGS-B',callback=trace,
                   options={'maxiter':maxiter,'gtol':1e-7,'ftol':1e-12,'maxls':40})
        val,gr=loss_grad(r.x,prep,penalty)
        finite=np.isfinite(val) and np.isfinite(r.x).all() and np.isfinite(gr).all()
        histories.append({'trajectory':trajectory,'final_coef':r.x.tolist(),'nfev':int(r.nfev),'start':name,'success':bool(r.success),'status':int(r.status),
                          'message':str(r.message),'nit':int(r.nit),'FIT_objective':val,
                          'gradient_inf':float(np.max(np.abs(gr))), 'finite':bool(finite)})
        if finite and (r.success or np.max(np.abs(gr))<1e-6):
            options.append((val,len(histories)-1,r.x.copy()))
    if not options:
        raise RuntimeError(f'no numerically converged FIT optimization: {histories}')
    _,chosen,coef=min(options,key=lambda t:(t[0],t[1]))
    return Model(np.asarray(scale,dtype=np.float64).copy(),coef),{
        'selected_start':histories[chosen]['start'],'starts':histories,'penalty':penalty,
        'temperature':1.0,'intercept':False,'source_split':source_split,
        'reference_denominators':prep[-1], 'ASR_updates':0, 'GPU_calls':0,
        'limitations':'local numerical optimum; no WER or generalization guarantee'}
