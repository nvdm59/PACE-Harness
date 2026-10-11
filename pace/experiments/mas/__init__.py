"""Process-verified multi-agent orchestration on CRMArena-Pro tasks.

A planner / data-worker / verifier system over the frozen CRMArena-Pro harness, whose process is
logged step by step, scored by a step-level verifier as well as by the outcome, and whose logged
(process, outcome) pairs train an orchestration reward model that feeds a priced controller.

    roles.py            PLANNER / WORKER / VERIFIER role calls, priced through the spend ledger
    orchestrator.py     the loop with PACE-style decision points {VERIFY, SKIP_VERIFY, REPLAN, ANSWER} x tier
    process_verify.py   outcome vs process rewards and the silent-failure analysis (offline)
    orm.py              orchestration reward model: process features -> P(correct), CV, export
    fake_backend.py     deterministic fake LLM + fake Salesforce for --dry-run and the tests
    run.py              CLI: run / analyze / estimate
"""
