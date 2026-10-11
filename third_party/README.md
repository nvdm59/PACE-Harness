# third_party/

External dependencies fetched at setup time. Nothing in this directory except this
note is committed (see `.gitignore`).

| path | what | license | fetched by |
|---|---|---|---|
| `CRMArena/` | [SalesforceAIResearch/CRMArena](https://github.com/SalesforceAIResearch/CRMArena), pinned to commit `6d84f3d` (CRMArena-Pro tasks, schemas, ReAct prompts, Salesforce connector, `Evaluator`) | CC BY-NC 4.0, research use only | `bash scripts/setup_crmarena.sh` |

`pace/harness/crmarena.py` imports CRMArena's modules from this checkout; no CRMArena code
is copied into PACE. The setup script also works around three upstream packaging issues:

- `requirements.txt` cannot be installed as pinned (`openai==1.51.2` conflicts with
  `litellm==1.67.2`, which needs `openai>=1.68.2`). The script installs it with the openai pin
  relaxed to `>=1.68.2,<2`.
- `crm_sandbox/` has no top-level `__init__.py`, so `pip install -e` maps no packages and
  `crm_sandbox` only imports from the repo root. The script adds a `.pth` entry for the checkout.
- `crm_sandbox/data/assets.py` imports Hugging Face `datasets`, which is not listed in
  CRMArena's requirements. The script installs it.

`CRMArena/.env` is created from CRMArena's `.env.example`, which carries the shared read-only
Salesforce org credentials that CRMArena publishes. You add `ANTHROPIC_API_KEY` yourself.
