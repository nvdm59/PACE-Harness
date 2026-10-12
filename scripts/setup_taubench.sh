#!/usr/bin/env bash
# Fetch τ-bench (Sierra, MIT license; github.com/sierra-research/tau-bench) as an external
# dependency into third_party/tau-bench (gitignored) and install it into the CRMArena venv.
#
# The harness uses only τ-bench's domain envs (tools, database, wiki, tasks), its LLM user
# simulator, message_to_action and Env.calculate_reward -- none of which import the provider SDKs
# in its install_requires (anthropic, mistralai, google-generativeai). The package is installed
# with --no-deps so those SDKs cannot move the venv's pinned litellm / openai / pydantic; the
# remaining imports (litellm, pydantic, tenacity) are already in .venv-crmarena. termcolor is
# added for τ-bench's own run.py. No new venv is needed.
#
# Usage:  bash scripts/setup_taubench.sh          (idempotent; run scripts/setup_crmarena.sh first)
#         source .venv-crmarena/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_URL="${TAUBENCH_REPO:-https://github.com/sierra-research/tau-bench}"
REF="${TAUBENCH_REF:-59a200c6d575d595120f1cb70fea53cef0632f6b}"   # commit inspected for this harness
DEST=third_party/tau-bench
VENV="${VENV:-.venv-crmarena}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "missing $VENV; run scripts/setup_crmarena.sh first (it creates the Python 3.12 venv with litellm)" >&2
  exit 1
fi
if [ ! -d "$DEST/.git" ]; then
  git clone "$REPO_URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$REF" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$REF" 2>/dev/null || echo "warning: could not pin $REF; using $(git -C "$DEST" rev-parse --short HEAD)"

"$VENV/bin/python" -m pip install --quiet --no-deps -e "$DEST"
"$VENV/bin/python" -m pip install --quiet termcolor
"$VENV/bin/python" -c "from tau_bench.envs import get_env; from tau_bench.envs.retail.tasks_test import TASKS_TEST; \
from tau_bench.envs.airline.tasks_test import TASKS; print(f'tau-bench ok: retail {len(TASKS_TEST)} / airline {len(TASKS)} test tasks')"

echo "τ-bench ready at $DEST ($(git -C "$DEST" rev-parse --short HEAD), $(head -1 "$DEST/LICENSE")); venv: $VENV"
echo "ANTHROPIC_API_KEY is read from the environment or third_party/CRMArena/.env"
echo "next: python -m pace.experiments.taubench.run collect --help"
