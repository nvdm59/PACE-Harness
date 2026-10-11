"""Continual MAS orchestrator: an online-trained decision point over a frozen multi-agent structure.

simenv.py          MAS simulator calibrated to the logged real run (results/mas/mas_b2b.jsonl), with the
                   real orchestrator's decision points, legal sets, feature layout and action names
drift.py           scripted drift events on a day schedule (harness "generations")
train.py           constrained GRPO on replay with the harness feature map, AWR+IPS, Monte Carlo eval
run_continual.py   the day loop: serve -> train a candidate -> DR gate -> promote / revert -> held-out eval
analyze.py         results/mas_continual/RESULTS.md from the run's JSONL logs
"""
