# Results-to-artifact map

| Manuscript item | Authoritative included assets | Entry |
|---|---|---|
| Main WER table and original strategies | `reproduction/data/INTEGRATED_RESULTS.csv`; `paper/results.tex` | `scripts/verify_release.py` |
| Four-system language WER and Local/Text increments | `reproduction/tables/LANGUAGE_ABSOLUTE_AND_DIFFERENCES.csv`; `paper/language_results.tex` | `reproduction/summarize_tables.py` |
| Original CV27 interval figure | `paper/figures/contrasts_data.json`; `figure2_rendered_values.json` | `paper/figures/contrasts_v2.py` |
| Architecture | `paper/figures/architecture_v2.py`, `synthetic_logmel_*` | supplied PDF/SVG |
| DupClean/Perm and half-FIT sensitivity, including harm | `APPENDED_CPU_RESULTS.csv`, `APPENDED_MACRO_RESULTS.csv`, `DIAGNOSTIC_MODELS.json` | retained values; no new fitting |
| Four existing appended paired diagnostics | `CV27_CLUSTER_COUNTS.npz`, `CPU_ANALYSIS.json`, `PAIRED_DIAGNOSTIC_INTERVALS.csv` | `portable/bootstrap_diagnostics.py` |
| End-to-end prototype timing and existing sampling intervals | `COST_TIMINGS_ANONYMOUS.csv`, `PAIRED_COST_COMPARISONS.csv`, `COST_PAIRED_UNCERTAINTY.csv` | `portable/cost_intervals.py` |
| All fitted selectors | `reproduction/models/*.json` | `portable/cache_pipeline.py`; original coefficients unchanged |
| WER-v3 | `reproduction/source/wer_rules_v3.py` | `reproduction/wer_score.py` |

The raw per-recording predictions needed to reconstruct every original interval
are not publicly bundled. Figure source data retain those intervals exactly;
anonymous counts support only the explicitly documented appended family. Do not
claim all original confidence intervals can be regenerated from this checkout.
