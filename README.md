# Matrix Arena — ARMLLM 2026 Entrance Competition

**Matrix Arena: Adversarial Inductive Matrix Completion** is the entrance-test
programming competition for the [Armenia LLM Summer School 2026](https://armllm.github.io/2026/competition).

You are given a sparse, partial view of a hidden structured matrix together with
row and column features, and you must reconstruct the hidden entries. In the
competitive arena you also choose which entries your opponent gets to observe —
without ever seeing the ground truth.

> **Key rule.** Participants do **not** create the ground-truth matrix. The
> grader creates one shared hidden matrix per seed. In a duel, both participants
> solve the *same* ground-truth matrix under different opponent-generated
> observation masks.

## Where to start

| Resource | Path |
|---|---|
| Competition kit (code, baselines, tests, grader) | [`matrix_arena/`](matrix_arena/) |
| Full kit README (API, scoring, tactics, how to run) | [`matrix_arena/README.md`](matrix_arena/README.md) |
| Formal problem statement | [`matrix_arena/challenge_statement.tex`](matrix_arena/challenge_statement.tex) · [PDF](matrix_arena/challenge_statement.pdf) |
| Copyable agent template | [`matrix_arena/examples/agent_template.py`](matrix_arena/examples/agent_template.py) |
| Reference baselines | [`matrix_arena/baselines/`](matrix_arena/baselines/) |

## Quick start

```bash
cd matrix_arena
pip install -e ".[dev]"        # core kit (numpy) + pytest/scipy
pytest -q                      # run the test suite

# score one or more agents on the public solve-only grader
python scripts/run_public_grader.py \
    --agents baselines/hybrid_agent.py baselines/ridge_agent.py --seeds 10

# run an arena tournament among agents
python scripts/run_tournament.py --agents baselines/*.py --mode full --seeds 5

# (optional) render match visualizations
pip install -e ".[viz]"
python scripts/visualize_match.py duel \
    --agent-a baselines/hybrid_agent.py --agent-b baselines/low_rank_agent.py \
    --seed 0 --budget small --output duel.png
```

## What you submit

A single Python file defining an `Agent` class with two methods:

```python
class Agent:
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        ...   # return the full (n, m) reconstruction

    def attack(self, X, Z, k, budget, seed):
        ...   # return a valid (n, n) observation mask for your opponent
```

Submit through the form linked on the
[competition page](https://armllm.github.io/2026/competition). **The final day for
submissions is July 5, 2026** (end of day, Yerevan time); you may resubmit until
then, and the latest valid submission counts.

## Questions

Please read [`matrix_arena/README.md`](matrix_arena/README.md) and the problem
statement first. If something is unclear or you think you've found a bug, open an
issue on this repository or email **armeniallm@gmail.com**.
