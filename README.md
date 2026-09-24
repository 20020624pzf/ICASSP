# Conditional Value of Permutation Scores in Frozen Multilingual ASR Reranking

Code, locked selector coefficients, aggregate results, anonymous counts, editable
figures, and manuscript sources for the author manuscript by **Zhifan Pan, Yuhui
Ming, Maihemuti Maimaiti, and Aishan Wumaier**. Zhifan Pan and Yuhui Ming contributed
equally; Maihemuti Maimaiti is the corresponding author. This repository does not
claim conference acceptance.

**Question:** Do permutation-derived scores add value beyond clean-audio
confidence, length, candidate consensus, and external text scores?

- Shared frozen Whisper-small candidate banks, five languages (ar/fa/pl/tr/it),
  three recognizer checkpoints, and one TRAIN fit per selector.
- Global improves over Conf by 0.1410 / 0.1824 WER percentage points on FLEURS / CV27.
- Text improves over Global by 0.9864 / 1.2043 points. Adding Global to Text has no
  interval-supported WER gain; this does not prove equivalence or redundancy.
- Local's incremental gain over Global and MWER's advantage over the matching
  ridge objective were not established. DEV reversals and adverse sensitivity
  results remain included.

## Start here

```bash
python3 scripts/verify_release.py
python3 reproduction/summarize_tables.py --out outputs/language_tables
```

These commands check the release and rebuild tables from existing aggregate
results. They do not run ASR, fit selectors, access held-out references, or launch
GPU jobs. [REPRODUCE.md](REPRODUCE.md) distinguishes tested offline operations from
inference APIs that require separately acquired data and exact model assets.

| Material | Location |
|---|---|
| Manuscript PDF and LaTeX | [paper/](paper/) |
| Selection, WER-v3, frozen inference and true token-masked PLL | [reproduction/](reproduction/) |
| Locked fitted coefficients | [reproduction/models/](reproduction/models/) |
| All retained aggregate results and timing rows | [reproduction/data/](reproduction/data/) |
| Language tables | [reproduction/tables/](reproduction/tables/) |
| Vector figures and editable source | [paper/figures/](paper/figures/) |
| Exact model/data/configuration identities | [reproduction/config/](reproduction/config/) |

See [DATA.md](DATA.md), [ENVIRONMENT.md](ENVIRONMENT.md),
[EXPERIMENTS.md](EXPERIMENTS.md), [RESULTS_MAP.md](RESULTS_MAP.md), and
[LIMITATIONS.md](LIMITATIONS.md). Source audio, corpus transcripts, original client
identifiers, private candidate/PLL caches, and model weights are not redistributed.
The displayed spectrogram is generated from a synthetic nonlinguistic signal.

This is a public, identified repository, not an anonymous review link. No project
wide open-source license is asserted; see [LICENSE_NOTICE.md](LICENSE_NOTICE.md).
Actual AI assistance is described in the manuscript and [PROVENANCE.md](PROVENANCE.md).
