#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=ICASSP2027_submission_final main.tex
