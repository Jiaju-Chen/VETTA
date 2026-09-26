# Reproducibility

This document describes what is included in the code snapshot and what must be
provided externally. The repository does not bundle benchmark data, model
weights, retrieval indexes, API credentials, or server-specific paths.

## Runtime

Training requires a Linux environment with CUDA, PyTorch, Ray, vLLM, and the
versions compatible with the inherited veRL/verl-agent stack. Install the
benchmark-specific environment dependencies as well; ALFWorld and WebShop have
additional environment requirements. The root `requirements.txt` is the broad
upstream development dependency set, not a pinned, self-contained lockfile.

Install the package in the prepared environment:

```bash
pip install -e . --no-deps
```

Use the project and benchmark installation instructions for the remaining
dependencies. Do not put credentials or local dataset paths in tracked files;
pass them through environment variables or local configuration.

## Data and checkpoints

- ALFWorld: provide the upstream environment assets and the prepared training
  and evaluation records expected by the `alfworld` runner.
- WebShop: provide the WebShop environment resources, product/search data, and
  prepared training and fixed-goal evaluation records.
- SearchQA: provide the processed training/evaluation records and a reachable
  retriever backed by the intended corpus/index. The retriever is a runtime
  dependency, not embedded data.
- Qwen checkpoints: provide the exact model revision used by the experiment.

Record the source/version and checksums of all downloaded assets in the local
run manifest. Keep data and checkpoints outside this repository.

## Focused tests

Run the tests for VETTA's advantage composition, value loss, action-format
metrics, validation aggregation, and the Turn-PPO comparison implementation:

```bash
pytest -q \
  tests/trainer/ppo/test_luna_unified_critic.py \
  tests/trainer/ppo/test_turn_ppo.py \
  tests/trainer/ppo/test_validation_metrics.py \
  tests/trainer/ppo/test_action_format_metrics.py \
  tests/trainer/ppo/test_advantage_case_study.py
```

These tests validate local tensor/aggregation behavior; they do not replace a
distributed rollout, checkpoint-resume, or full benchmark evaluation. Before a
paper release, run the same tests in the pinned CUDA environment, then execute
one small end-to-end rollout for each benchmark and archive the resolved config,
code revision, data manifest, and checkpoint selection rule.

## Method configuration

The trainer configuration is in `verl/trainer/config/ppo_trainer.yaml`. The
VETTA critic/advantage path is selected with `algorithm.adv_estimator=luna_unified`
and configured through `algorithm.hybrid_advantage.*` and `critic.*`. Token and
turn discount/lambda values, residual scale, critic depth, head count,
whitening, rollout length, and evaluation protocol are experiment parameters;
set and record them explicitly for every run. The implementation defaults are
not a substitute for a resolved experiment config.

## Release hygiene

Before publishing a code release, scan source files and packaged assets for
credentials, machine-specific paths, experiment logs, and data that cannot be
redistributed. Verify the documented setup from a clean checkout and include
only the assets required for the supported workflows.
