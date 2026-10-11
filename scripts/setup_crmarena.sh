#!/usr/bin/env bash
# Fetch CRMArena (Salesforce AI Research, CC BY-NC 4.0, research use) as an
# external dependency into third_party/CRMArena (gitignored), install it
# editable into a Python 3.12 venv, and seed its .env from .env.example.
#
# CRMArena pins numpy==2.0.1 / pandas==2.2.2 / pydantic==2.8.2, which do not
# build on Python 3.13+, so this uses python3.12 by default (override: PYTHON=...).
#
# Usage:  bash scripts/setup_crmarena.sh          (idempotent)
#         source .venv-crmarena/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_URL="${CRMARENA_REPO:-https://github.com/SalesforceAIResearch/CRMArena}"
REF="${CRMARENA_REF:-6d84f3d71305af0fd3d5ed3c1936b7887464455a}"   # commit inspected for this harness
DEST=third_party/CRMArena
VENV="${VENV:-.venv-crmarena}"
PY="${PYTHON:-python3.12}"

if [ ! -d "$DEST/.git" ]; then
  git clone "$REPO_URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$REF" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$REF" 2>/dev/null || echo "warning: could not pin $REF; using $(git -C "$DEST" rev-parse --short HEAD)"

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
# CRMArena's requirements.txt is unsatisfiable as pinned (openai==1.51.2 vs
# litellm==1.67.2, which needs openai>=1.68.2). Install the package without deps,
# then its requirements with only the openai pin relaxed.
"$VENV/bin/python" -m pip install --quiet --no-deps -e "$DEST"
grep -v '^openai==' "$DEST/requirements.txt" > "$VENV/crmarena-requirements.txt"
echo 'openai>=1.68.2,<2' >> "$VENV/crmarena-requirements.txt"
"$VENV/bin/python" -m pip install --quiet -r "$VENV/crmarena-requirements.txt"
# crm_sandbox/ has no top-level __init__.py, so setup.py's find_packages() maps
# nothing and the editable install cannot import it outside the repo root.
# Put the checkout on the venv's path instead.
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$(cd "$DEST" && pwd)" > "$SITE/crmarena_checkout.pth"
# crm_sandbox/data/assets.py imports `datasets` (Hugging Face) but CRMArena's
# requirements.txt does not list it; PACE itself needs numpy (already pulled in).
"$VENV/bin/python" -m pip install --quiet datasets pytest

if [ ! -f "$DEST/.env" ]; then
  cp "$DEST/.env.example" "$DEST/.env"
  {
    echo ""
    echo "# --- added by PACE scripts/setup_crmarena.sh ---"
    echo "# LLM key for the agent tiers and the answer extractor (use a dedicated, spend-limited workspace):"
    echo "ANTHROPIC_API_KEY="
  } >> "$DEST/.env"
  echo "created $DEST/.env (Salesforce org creds from CRMArena's .env.example; fill ANTHROPIC_API_KEY)"
else
  echo "$DEST/.env exists; left unchanged"
fi

echo "CRMArena ready at $DEST ($(git -C "$DEST" rev-parse --short HEAD)); venv: $VENV"
echo "next: source $VENV/bin/activate && python -m pace.collect --harness crmarena --help"
