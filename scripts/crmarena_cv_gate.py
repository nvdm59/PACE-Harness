"""Cross-fitted DR gate on the CRMArena-Pro logs (B2B 300 + B2C 150): train on training folds,
gate on the held-out fold's realised one-arm log. See crmarena_offline.cv_gate_section.

  python scripts/crmarena_cv_gate.py                 # both orgs, a few minutes on a laptop CPU, deterministic

Writes results/crmarena_300/cv_gate.{json,md} and results/crmarena_b2c/cv_gate.{json,md}
(new files; the committed gate section of tables.md / results.json is untouched).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import crmarena_offline as off  # noqa: E402
from crmarena_table1 import ORGS, load_org  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orgs", nargs="*", default=["b2b", "b2c"], choices=list(ORGS))
    ap.add_argument("--redraws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(argv)
    for org in a.orgs:
        eps, T, folds = load_org(org)
        res, lines = off.cv_gate_section(eps, T, folds, a.redraws, a.boot)
        out = ORGS[org]["out"]
        res = {"org": org, "n_tasks": len(T), **res}
        (out / "cv_gate.json").write_text(json.dumps(res, indent=1))
        md = "\n".join([f"## Cross-fitted DR gate: CRMArena-Pro {org.upper()}, {len(T)} tasks", ""] + lines) + "\n"
        (out / "cv_gate.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
