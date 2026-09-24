# Environments

The release verifier and CSV table formatter require only Python's standard
library. NumPy 2.2.6 is pinned for optional cached selectors and existing bootstrap
reproduction. No dependency is installed automatically by the check command.

Recorded scientific environment (not a promise that arbitrary versions match):
PyTorch 2.4.0+cu121, Transformers 4.57.6, PEFT 0.18.1, NumPy 2.2.6, SoundFile
0.13.1, SciPy 1.15.3, RapidFuzz 3.14.5, Numba 0.65.0, safetensors 0.7.0,
SentencePiece 0.2.1. See `reproduction/portable/LOCAL_ENVIRONMENT_METADATA.json`.
Historical inference used NVIDIA A40 GPUs with recorded shared load. Manuscript
compilation uses latexmk/pdfLaTeX; vector generation uses CairoSVG.

Frozen inference dependencies and trained adapters are intentionally separate
from the minimal CPU environment. The public package has not undergone a fresh
full-corpus GPU reconstruction. CPU cache-processing time is not deployment cost.
