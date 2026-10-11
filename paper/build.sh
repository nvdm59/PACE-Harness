#!/usr/bin/env bash
# Regenerate figures from the committed result files, then compile main.pdf with tectonic.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv-crmarena/bin/python}"
TECTONIC="${TECTONIC:-/opt/homebrew/bin/tectonic}"
"$PY" make_figures.py
"$TECTONIC" main.tex
echo "built $(pwd)/main.pdf"
