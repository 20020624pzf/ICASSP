"""CPU prototype: reference-free inference after TRAIN-only signed-edit-gain fitting.
This is a generic regularized regression utility, not a verified ASR implementation
and not a novelty claim. Runtime audio features must be computed separately.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class GainModel:
    scale: np.ndarray
    coef: np.ndarray
    feature_names: tuple[str, ...]

    def predict_gain(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.coef) or not np.isfinite(x).all():
            raise ValueError('Invalid feature matrix')
        return (x / self.scale) @ self.coef

    def choose(self, pair_features: np.ndarray, beam_index: int) -> int:
        """pair_features[k] = features(candidate k relative to beam top1).
        No references, error counts, TEST labels, or oracle inputs are accepted.
        """
        x = np.asarray(pair_features, dtype=np.float64)
        if not 0 <= beam_index < len(x): raise ValueError('Invalid beam index')
        if not np.allclose(x[beam_index],0,atol=1e-12,rtol=0):
            raise ValueError('Beam reference row must be zero')
        gain = self.predict_gain(x)
        winner = int(np.argmax(gain))
        return winner if gain[winner] > 0 else beam_index


def fit_gain_model(x, target_edit_gain, sample_weight, feature_names,
                   *, source_split: str, penalty: float = 0.01) -> GainModel:
    if source_split != 'TRAIN_CAL_FIT':
        raise ValueError('This frozen protocol fits on TRAIN_CAL_FIT only')
    x=np.asarray(x,dtype=np.float64);y=np.asarray(target_edit_gain,dtype=np.float64)
    w=np.asarray(sample_weight,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=(len(x),) or x.shape[1]!=len(feature_names):
        raise ValueError('Shape mismatch')
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError('Nonfinite data')
    if len(x)==0 or np.any(w<0) or w.sum()<=0 or not np.isfinite(penalty) or penalty<=0:
        raise ValueError('Invalid data/regularization')
    w=w/w.sum()
    scale=np.sqrt(np.sum(w[:,None]*x*x,axis=0));scale=np.where(scale>1e-12,scale,1.)
    z=x/scale
    coef=np.linalg.solve(z.T@(w[:,None]*z)+penalty*np.eye(z.shape[1]),z.T@(w*y))
    return GainModel(scale,coef,tuple(feature_names))


def reference_dependent_oracle(error_counts: np.ndarray) -> np.ndarray:
    """Diagnostic only: never call in production candidate selection."""
    e=np.asarray(error_counts)
    if e.ndim!=2 or not np.isfinite(e).all() or np.any(e<0):
        raise ValueError('Invalid diagnostic errors')
    return np.min(e,axis=1)
