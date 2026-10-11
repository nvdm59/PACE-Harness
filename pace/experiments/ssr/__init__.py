"""Priced Socratic self-refinement (SSR) as a test-time compute policy on CRMArena-Pro.

refine.py   one SSR round: decompose -> re-solve (SOQL through the harness) -> confidence -> refine
policy.py   refinement policies (never / always / uncertainty-gated / PACE controller) + training
fake.py     deterministic fake backend for --dry-run (no API, no Salesforce)
run.py      CLI: `run` (collect a refinement tree or serve one policy) and `analyze`
analyze.py  exact policy evaluation on the logged tree, calibration, harm, frontier, RESULTS.md
"""
