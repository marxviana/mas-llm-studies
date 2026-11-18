# mas-llm-studies

Code, data and LLM transcripts for the article

> M. Viana, E. Cirilo, P. Alencar, D. Cowan and C. Lucena,
> **"Do Large Language Models Improve Adaptive Normative Agents? Norm Adaptation and Norm Compliance in Paired Simulations"** (submitted to *IEEE Access*).

The repository contains the `llm-mas` Python platform. The platform implements adaptive normative agents whose components can be either a large language model (LLM) or a deterministic counterpart that receives the same information. It holds the three controlled studies of the article:

| Study | Question | Command | Results |
|---|---|---|---|
| 1 | Does an LLM improve negotiation, norm recommendation and alarm triage? (RQ1–RQ4) | `llm-mas compare-llm` | `results/v3/` |
| 1 (ext.) | Evidence labeled by provenance (RQ8); LLM validator in isolation (RQ9) | `llm-mas extension --parts provenance validator` | `results/ext-llm/` |
| 2 | Norm deviation versus norm adaptation; LLM deliberation under enforcement (RQ5–RQ7) | `llm-mas deliberation` | `results/phase2-offline/`, `results/phase2-llm/`, `results/phase2-llm-s2p1/` |
| 3 | LLM judgment elicited apart from incentives, decided by an expected-value rule (RQ10) | `llm-mas extension --parts study3` | `results/ext-llm/` |

Study 2 was first reported in the workshop paper *"Trustworthy LLM Agent Autonomy Through Normative Governance: Norm Deviation versus Norm Adaptation"*. That paper's repository is [marxviana/mas-anaml-llm](https://github.com/marxviana/mas-anaml-llm). The follow-up metamodel, ANA-ML 3.0, is in [marxviana/anaml3-metamodel](https://github.com/marxviana/anaml3-metamodel).

## Layout

```
src/llm_mas/
├── core/        agents with a MAPE-K loop, environment, synchronous simulation
├── llm/         Claude client with a cost cap, test doubles, action parser
├── norms/       norms, norm knowledge base, monitor (sanctions, rewards, regimented
│                and regulated enforcement), LLM recommender/validator/auditor,
│                norm adaptation engine with deterministic safety bounds
├── scenarios/   marketplace (book trading), icu (intensive-care alarms), project
├── experiments.py, experiments_phase2.py, experiments_ext.py   the three studies
└── cli.py
analysis/        scripts that extract every number, table and data figure of the article
results/         raw results (results.json), reports and full LLM transcripts (transcripts.jsonl)
tests/           110 tests (order invariance, paired seeds, monitor, components)
```

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # offline: no API key needed
.venv/bin/pip install -e ".[llm,dev]"      # to call Claude
.venv/bin/pytest
```

## Reproducing the article without API cost

Every LLM component has a deterministic counterpart, and every LLM decision is stored in the transcripts. The analysis scripts therefore regenerate the article's numbers from `results/` without any API call:

```bash
PYTHONPATH=src .venv/bin/python analysis/data.py            # Study 1 (results/v3)
PYTHONPATH=src .venv/bin/python analysis/data_phase2.py     # Study 2
PYTHONPATH=src .venv/bin/python analysis/justifications.py  # Study 2: replay of the LLM's decisions and beliefs
PYTHONPATH=src .venv/bin/python analysis/recalibration.py results/ext-llm   # Study 3: post hoc recalibration
```

The offline conditions can be rerun directly, for example `llm-mas deliberation --seeds 0 1 2` or `llm-mas icu --compare --seeds 0 1 2`.

## Rerunning the LLM conditions (paid)

The LLM runs used `claude-opus-5-5` with effort `low` and seeds 0, 1 and 2. Use `--dry-run` first: it estimates the number of calls and the cost without calling the API. `--max-cost` stops a run at a budget.

| Run | Command | Calls | Est. cost (US$) |
|---|---|---|---|
| `results/v3` | `llm-mas compare-llm --llm claude --effort low --seeds 0 1 2 --llm-decision --out results/v3` | 272 | 4.73 |
| `results/phase2-llm` (+ `-s2p1`) | `llm-mas deliberation --llm claude --effort low --seeds 0 1 2 --out results/phase2-llm` | 930 | 17.55 |
| `results/ext-llm` | `llm-mas extension --llm claude --effort low --seeds 0 1 2 --out results/ext-llm` | 406 | 7.22 |

The `phase2-llm` run stopped at its US$ 16 cap before its ninth run. That run (seed 2, p = 1.0) was completed separately in `phase2-llm-s2p1`.

The API key goes in `.env` (see `.env.example`).
