#!/usr/bin/env bash
# PACE v0 end to end on the simulated harness (target: < 2 minutes on a laptop).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
BUDGET=0.30
REPRICE="mid:1.2"
mkdir -p traces policies results
t0=$(date +%s)

echo "== 1. collect 3000 episodes on sim v1 with a uniform behavior policy"
$PY -m pace.collect --harness v1 --policy uniform --n 3000 --seed 1 --out traces/v1_behavior.jsonl

echo "== 2. constrained GRPO on replay, budget $BUDGET (learned lambda)"
$PY -m pace.train_offline --traces traces/v1_behavior.jsonl --method cgrpo --budget $BUDGET \
    --seed 2 --out policies/policy_v1.json --log results/train_v1.jsonl

echo "== 3. gate: candidate vs behavior, on the v1 log"
$PY -m pace.gate --traces traces/v1_behavior.jsonl --candidate policies/policy_v1.json \
    --incumbent behavior --budget $BUDGET --title "Gate v1: cgrpo candidate vs behavior" \
    --json-out results/gate_v1.json

echo "== 4. drift: harness v1 -> v2 (+CACHED_RETRIEVE, aged verifier, frontier x1.5) + reprice $REPRICE"
echo "== 5. collect 6000 fresh v2 episodes, serving the frozen policy with eps=0.3 exploration"
$PY -m pace.collect --harness v2 --reprice $REPRICE --policy policies/policy_v1.json --eps 0.3 \
    --n 6000 --seed 3 --out traces/v2_frozen.jsonl

echo "== 6. retrain on v2 traces (warm start from the frozen policy)"
$PY -m pace.train_offline --traces traces/v2_frozen.jsonl --method cgrpo --budget $BUDGET \
    --reprice $REPRICE --init policies/policy_v1.json --seed 4 \
    --out policies/policy_v2.json --log results/train_v2.jsonl

echo "== 7. gate: adapted vs frozen, on the v2 log (quality bar = frozen's DR quality at its v1 promotion)"
$PY -m pace.gate --traces traces/v2_frozen.jsonl --candidate policies/policy_v2.json \
    --incumbent policies/policy_v1.json --q-prev-from results/gate_v1.json --budget $BUDGET --title "Gate v2: adapted vs frozen" \
    --json-out results/gate_v2.json

echo "== 8. final table"
$PY -m pace.drift --frozen policies/policy_v1.json --adapted policies/policy_v2.json \
    --gate-json results/gate_v2.json --reprice $REPRICE --budget $BUDGET --json-out results/drift.json

echo "done in $(( $(date +%s) - t0 ))s"
