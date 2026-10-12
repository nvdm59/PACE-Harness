#!/usr/bin/env bash
# The run_v0.sh simulator pipeline repeated over replicate seeds.
# Replicate s uses component seeds 10s+1 .. 10s+4 for collect/train/collect/train
# (s=0 reproduces run_v0.sh exactly) and seed s for the gate bootstrap and the drift evaluation.
# Usage: scripts/run_sim_seeds.sh [seeds...]   (default 0 1 2 3 4)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
BUDGET=0.30
REPRICE="mid:1.2"
SEEDS="${*:-0 1 2 3 4}"
TR="${SIM_TRACES_DIR:-traces/sim_seeds}"
for s in $SEEDS; do
  O=results/sim_seeds/seed$s; T=$TR/seed$s; mkdir -p "$O" "$T"
  b=$((10*s))
  $PY -m pace.collect --harness v1 --policy uniform --n 3000 --seed $((b+1)) --out $T/v1_behavior.jsonl
  $PY -m pace.train_offline --traces $T/v1_behavior.jsonl --method cgrpo --budget $BUDGET \
      --seed $((b+2)) --out $O/policy_v1.json --log $O/train_v1.jsonl
  $PY -m pace.gate --traces $T/v1_behavior.jsonl --candidate $O/policy_v1.json --incumbent behavior \
      --budget $BUDGET --seed $s --title "Gate v1 (seed $s)" --json-out $O/gate_v1.json
  $PY -m pace.collect --harness v2 --reprice $REPRICE --policy $O/policy_v1.json --eps 0.3 \
      --n 6000 --seed $((b+3)) --out $T/v2_frozen.jsonl
  $PY -m pace.train_offline --traces $T/v2_frozen.jsonl --method cgrpo --budget $BUDGET \
      --reprice $REPRICE --init $O/policy_v1.json --seed $((b+4)) --out $O/policy_v2.json --log $O/train_v2.jsonl
  $PY -m pace.gate --traces $T/v2_frozen.jsonl --candidate $O/policy_v2.json --incumbent $O/policy_v1.json \
      --q-prev-from $O/gate_v1.json --budget $BUDGET --seed $s --title "Gate v2 (seed $s)" --json-out $O/gate_v2.json
  $PY -m pace.drift --frozen $O/policy_v1.json --adapted $O/policy_v2.json --gate-json $O/gate_v2.json \
      --reprice $REPRICE --budget $BUDGET --seed $s --json-out $O/drift.json
done
