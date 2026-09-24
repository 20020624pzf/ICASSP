# Reproduction entry

## One-command release check (no model dependencies)

From the repository root:

```bash
python3 scripts/verify_release.py
```

This validates the file manifest, frozen coefficient dimensions and scientific
configuration, and confirms key paper values from existing per-language CSV rows.
It is a package check, not new performance evidence.

## Rebuild language tables from existing results

```bash
python3 reproduction/summarize_tables.py --out outputs/language_tables
```

The command preserves all original CSV values and formats 40 absolute WER cells
and 40 language differences. It does not decode, refit, or resample.

## Compile the manuscript

Install a LaTeX distribution with latexmk, pdfLaTeX, and the standard packages used
by the supplied source. The conference style is included unchanged.

```bash
bash paper/build.sh
```

Expected output: `paper/ICASSP2027_submission_final.pdf`, five US Letter pages.
Pre-rendered vector figures are included, so the paper build does not need Python
or GNU Make. `make -C paper` remains an optional equivalent if Make is installed.

## Optional reproduction of already reported uncertainty (CPU)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r reproduction/requirements-cached-select.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproduction/portable/bootstrap_diagnostics.py --out outputs/reproduced_intervals.json
python reproduction/portable/cost_intervals.py --out outputs/reproduced_cost
```

Create `outputs/` first if no preceding command has done so. These are the existing
locked estimators, not new statistical families: 10,000 common client-bootstrap
draws, seed 1729, for the four appended CV27 diagnostics (98.75% adjusted and 95%
intervals). The timing script reproduces existing paired recording sampling
intervals. Neither estimates uncertainty over the full development history or
over shared-load processes. No new intervals were generated to publish this repo.

## Apply locked selectors to legally available local caches

Complete `reproduction/portable/spec.example.json` with real file hashes and
paths. Feature files, local reference files, and the exact-text PLL SQLite cache
are separate; none is hosted here. Paths resolve relative to the spec file.

```bash
python reproduction/portable/cache_pipeline.py select --spec /authorized/panel.json --out outputs/choices.jsonl
python reproduction/portable/cache_pipeline.py evaluate --spec /authorized/panel.json --choices outputs/choices.jsonl --out outputs/wer.json
```

Selection never reads reference content; evaluation verifies the locked choices
before scoring complete references with WER-v3. Authorized five-language
reference/hypothesis pairs can also be scored using `reproduction/wer_score.py`.
The original TRAIN-FIT reconstruction entry (`cache_pipeline.py fit`) is supplied
for methodological transparency, not needed to apply the included coefficients.
Do not select new fits or rules using CHECK/DEV/TEST.

## Reconstruct candidate features (assets required; no full rerun claimed)

1. Obtain the exact data versions and local checkpoint files in `DATA.md` and
   `reproduction/config/VERSIONS_AND_ASSETS.json`. The trained adapters are not
   included, so full reproduction is unavailable from this checkout alone.
2. `portable/prepare_inputs.py` accepts a JSON configuration containing `split`
   and five language entries (`tsv`, `tsv_sha256`, `audio_dir`), keeping references
   separate from inference manifests.
3. Complete `portable/config.example.json` and set `ASR_REPRO_CONFIG` to it. In a
   worker legitimately authorized by your compute site, use the API in
   `portable/frozen_inference.py`: load the frozen ASR, `generate` in modes `G`
   and `N5`, then `acoustic_features`; load the frozen MLM and `score_pll` once per
   exact retained text. These APIs do not launch or authorize workers.
4. Use the existing locked coefficients and the reference-free selection command
   above. Live CPU/GPU ASR/MLM reconstruction has not been validated by this release.

The original model coefficients, effective decoder configurations and PLL identity
are byte-preserved. Historical absolute paths inside those JSON provenance records
are inert source locations, not defaults to use on a different installation.
In particular, `PLL_IDENTITY.json` is preserved because its exact byte hash binds
existing caches. Do not edit it to point to new local files.

## Figures

The exact PDF/SVG files used by the manuscript are supplied. To regenerate their
layout, install `paper/requirements-figures.txt` and run the two `*_v2.py` scripts
under `paper/figures/`. The synthetic mel generator has a separate requirements
file and provenance record. Figure 2 reads existing numbers and intervals; these
commands neither add statistical tests nor change the evaluated systems.
