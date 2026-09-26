# Getting Started

This is an operator guide for the local VETTA snapshot. It gives the shortest
path from a clean checkout to environment checks and training commands. Exact
results still depend on the model revision, benchmark assets, GPU topology, and
resolved Hydra configuration recorded for each run.

## 1. Runtime setup

Use Linux with NVIDIA GPUs and a CUDA/PyTorch combination supported by the
installed vLLM and FlashAttention builds. Python 3.10 is the safest common
choice for the bundled ALFWorld and WebShop integrations. The repository's
`requirements.txt` is an upstream development list, not a fully pinned lockfile;
do not assume that installing it alone recreates a research run.

Example environment creation (adjust the PyTorch CUDA wheel to the host driver):

```bash
conda create -n vetta python=3.10 -y
conda activate vetta
python -m pip install --upgrade pip wheel
# Install the PyTorch build matching this host's CUDA driver, then:
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Install a compatible vLLM and FlashAttention build after PyTorch. The inherited
environment guide currently documents PyTorch 2.6.0/CUDA 12.4 and vLLM 0.8.x;
those versions must be matched to the target machine rather than blindly
combined. Save the actual versions with every run:

```bash
python - <<'PY'
import torch, transformers, ray
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("ray", ray.__version__)
try:
    import vllm
    print("vllm", vllm.__version__)
except ImportError as exc:
    print("vllm missing:", exc)
PY
nvidia-smi
```

For a reproducible paper run, replace this example setup with an environment
lock file tied to the exact code revision. No such tested lock file is included
yet.

## 2. Benchmark assets

Keep downloaded assets outside the Git checkout and record their source,
version, and checksum in a run manifest.

### ALFWorld

Install the environment package and download its game assets:

```bash
python -m pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
alfworld-play-tw
```

The final command should open an interactive TextWorld game. Training expects
the ALFWorld `valid_seen` data for the seen-140 protocol; the environment's
PDDL/game assets are not model weights and are not stored in this repository.
Set the `ALFWORLD_DATA` location according to the installed ALFWorld config and
verify that the train and `json_2.1.1/valid_seen` trees exist before launching.

### WebShop

Install the bundled WebShop fork and its data in the same Python environment:

```bash
cd agent_system/environments/env_package/webshop/webshop
bash setup.sh -d all
cd ../../../../..
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX}"
export PATH="$JAVA_HOME/bin:$PATH"
test -x "$JAVA_HOME/bin/javac"
python agent_system/environments/env_package/webshop/webshop/run_web_agent_text_env.py
```

The final command should start the local text environment. WebShop setup may
need access to the upstream Google Drive assets; follow the upstream setup
instructions if a download is blocked. The formal endpoint uses the fixed 500
goal IDs `0..499`; changing `env.seed` changes goal construction and is not a
valid decoding-seed replication. The PPO launchers accept `TRAIN_DATA_FILE` and
`VAL_DATA_FILE` to select audited parquet files; set `SKIP_DATA_PREP=1` when
using those files so the example-data generator does not overwrite them.

### SearchQA

SearchQA requires processed train/eval parquet files, the intended corpus and
FAISS index, and a reachable retrieval service. The retrieval server included
here defaults to E5-base-v2, Wiki-18, and top-k 3. Set the index/corpus paths to
the exact intended artifacts before starting it:

```bash
python examples/search/retriever/retrieval_server.py \
  --index_path "$SEARCHQA_INDEX" \
  --corpus_path "$SEARCHQA_CORPUS" \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --topk 3 \
  --faiss_gpu \
  --port 8000
```

In another shell, check the service:

```bash
curl -fsS http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is an example question?","topk":3}'
```

This snapshot does not include the processed SearchQA dataset or a canonical
SearchQA train launcher. Do not treat the retriever check as a training smoke
test; use the dataset-specific launcher and manifest for SearchQA.

## 3. Code and environment smoke checks

Run these before requesting GPU allocation:

```bash
python -m compileall -q verl agent_system
bash -n examples/ppo_trainer/run_alfworld.sh
bash -n examples/ppo_trainer/run_webshop.sh
pytest -q \
  tests/trainer/ppo/test_luna_unified_critic.py \
  tests/trainer/ppo/test_turn_ppo.py \
  tests/trainer/ppo/test_validation_metrics.py \
  tests/trainer/ppo/test_action_format_metrics.py \
  tests/trainer/ppo/test_advantage_case_study.py
```

For a GPU smoke, use a disposable HOME/output/Ray directory, a small batch, and
one update. This WebShop example exercises one rollout and one VETTA update,
without validation, W&B logging, or a formal checkpoint:

```bash
SMOKE_ROOT=/tmp/vetta-webshop-smoke
mkdir -p "$SMOKE_ROOT/home"
HOME="$SMOKE_ROOT/home" TRAIN_DATA_SIZE=8 VAL_DATA_SIZE=8 \
bash examples/ppo_trainer/run_webshop.sh vllm \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  algorithm.adv_estimator=luna_unified \
  critic.model.num_value_heads=2 \
  +critic.model.override_config.num_hidden_layers=2 \
  algorithm.gamma=1.0 algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=0.95 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.composition_mode=residual \
  algorithm.hybrid_advantage.token_residual_scale=3.0 \
  algorithm.hybrid_advantage.whiten_advantages=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  critic.ppo_mini_batch_size=8 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  env.max_steps=3 \
  trainer.total_epochs=1 trainer.test_freq=-1 trainer.save_freq=-1 \
  trainer.val_before_train=False trainer.resume_mode=disable \
  "trainer.logger=[console]" trainer.project_name=vetta-smoke \
  trainer.experiment_name=webshop-one-update \
  trainer.default_local_dir="$SMOKE_ROOT/checkpoints" \
  trainer.n_gpus_per_node=8 \
  +ray_init._temp_dir=/tmp/vetta-smoke-ray
```

The launcher regenerates only the example parquet under the disposable HOME.
Keep Ray's temporary path short: Ray creates Unix socket paths with a strict
length limit, so do not place its session directory under a long project path.
For an ALFWorld smoke, use its data path/asset setup and keep the same isolated
output-directory rule. A successful Python compile is not evidence that Ray,
vLLM, the environment, or checkpoint resume works; save the smoke log and
resolved config.

## 4. Launch commands

Set a local model path first. These commands use the bundled environment
launchers, which prepare their example text parquet files. For a paper run,
replace those files with the audited benchmark-specific parquet files and set
the resolved batch sizes and evaluation manifest explicitly.

```bash
export MODEL_PATH=/models/Qwen2.5-1.5B-Instruct
```

### Standard token-level PPO

The standard PPO baseline uses token GAE (`gae`), a scalar full-depth Critic,
and the default token-level clipped policy ratio:

```bash
bash examples/ppo_trainer/run_webshop.sh vllm \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  algorithm.adv_estimator=gae \
  critic.model.num_value_heads=1 \
  env.seed=0 \
  trainer.n_gpus_per_node=8 \
  trainer.experiment_name=webshop-token-ppo
```

For the cross-observation token baseline used in this project, change the
estimator to `sao_skip_observation`; this is not the same as ordinary response
GAE:

```bash
  algorithm.adv_estimator=sao_skip_observation
```

Use that override in the full command, not as a second standalone command.

### Turn-PPO

Turn-PPO needs both the turn-level GAE estimator and the response-level actor
ratio/clipping objective. Setting only one of these is invalid; the trainer
checks that they agree.

```bash
bash examples/ppo_trainer/run_webshop.sh vllm \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  algorithm.adv_estimator=turn_ppo \
  algorithm.gamma=0.95 algorithm.lam=0.95 \
  actor_rollout_ref.actor.policy_loss.loss_mode=turn_ppo \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  critic.model.num_value_heads=1 \
  env.seed=0 \
  trainer.n_gpus_per_node=8 \
  trainer.experiment_name=webshop-turn-ppo
```

This is distinct from turn-GAE plus token-level PPO ratio (`turn_gae_token_ratio`)
and from the VETTA `turn_only` actor-credit ablation.

### VETTA shared dual-head Critic

This uses a shallow shared Critic, two value heads, residual advantage
composition, and token-level PPO ratio:

```bash
bash examples/ppo_trainer/run_webshop.sh vllm \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  algorithm.adv_estimator=luna_unified \
  critic.model.num_value_heads=2 \
  +critic.model.override_config.num_hidden_layers=2 \
  critic.unified_luna_turn_loss_coef=1.0 \
  algorithm.gamma=1.0 algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=0.95 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.composition_mode=residual \
  algorithm.hybrid_advantage.token_residual_scale=3.0 \
  algorithm.hybrid_advantage.whiten_advantages=True \
  env.seed=0 \
  trainer.n_gpus_per_node=8 \
  trainer.experiment_name=webshop-vetta
```

The WebShop command above uses its best-observed `alpha=3`. ALFWorld's
best-observed row uses `alpha=1`; do not reuse the WebShop override. With
benchmark-prepared parquet files, the ALFWorld launch is:

```bash
export ALFWORLD_DATA=/path/to/alfworld-data
MODEL_PATH=/models/Qwen2.5-1.5B-Instruct \
TRAIN_DATA_FILE=/data/alfworld/train.parquet \
VAL_DATA_FILE=/data/alfworld/valid_seen_140.parquet \
SKIP_DATA_PREP=1 TRAIN_DATA_SIZE=128 VAL_DATA_SIZE=140 \
bash examples/ppo_trainer/run_alfworld.sh vllm \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  algorithm.adv_estimator=luna_unified \
  critic.model.num_value_heads=2 \
  +critic.model.override_config.num_hidden_layers=2 \
  critic.unified_luna_turn_loss_coef=1.0 \
  algorithm.gamma=1.0 algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=0.95 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.composition_mode=residual \
  algorithm.hybrid_advantage.token_residual_scale=1.0 \
  algorithm.hybrid_advantage.whiten_advantages=True \
  env.seed=0 env.alfworld.eval_dataset=eval_in_distribution \
  trainer.n_gpus_per_node=8 \
  trainer.experiment_name=alfworld-vetta-alpha1
```

These are launch templates, not substitutes for recording the resolved Hydra
configuration, data manifest, and exact checkpoint selection for a paper run.

## 5. Best-observed settings

The following are the best-observed VETTA candidates in the current experiment
ledger, not universal optima. The ALFWorld/WebShop rows are 1.5B runs; SearchQA
is listed at 7B because that is the strongest recorded SearchQA endpoint in the
current ledger. Recheck the endpoint and protocol before using the numbers in a
paper table.

| Benchmark / model | Critic | Actor credit | Token GAE | Turn GAE | Data / endpoint |
|---|---|---|---|---|---|
| ALFWorld 1.5B | Shared 2-layer, 2 heads | Residual, alpha=1 | gamma/lambda = 1/1 | gamma/lambda = 0.95/0.95 | 150 updates; seen-140; `env.seed=0`; decoding 123/456/789: 91.90% ± 1.09 pp |
| WebShop 1.5B | Shared 2-layer, 2 heads | Residual, alpha=3 | gamma/lambda = 1/1 | gamma/lambda = 0.95/0.95 | 150 updates; fixed goals `0..499`; `env.seed=0`; decoding 123/456/789: 73.4% ± 1.04 pp, task score 0.8391 ± 0.0083 |
| SearchQA 7B | Shared 8-layer, 2 heads | Residual, alpha=1 | gamma/lambda = 1/1 | gamma/lambda = 0.95/0.95 | 200 updates; full 51,713-case test; seven-task arithmetic average 44.43% (greedy) |

For ALFWorld and WebShop, the decoding seed changes only generation randomness;
do not change the environment seed or task manifest. WebShop validation must
retain the same ordered goal IDs `0..499`. SearchQA's official full-test
evaluation in the recorded run used greedy decoding; do not substitute its
training monitor batch for the full endpoint.

The ± values for ALFWorld and WebShop are variation across decoding seeds for
one trained checkpoint, not variation across training seeds. These values are
configuration references, not yet a complete public run manifest. Optimizer
micro-batches, GPU count, tokenizer/model revision, corpus checksum,
prompt/response limits, checkpoint selection, and all resolved Hydra values
must be copied from the exact run launcher before claiming a reproduction.

## 6. W&B logging

Use an interactive login on the training host, or provide credentials through
the host's secret manager. Never put a W&B key in a launcher, README, shell
history, or committed `.netrc`:

```bash
wandb login
wandb status
```

Set `trainer.logger` to include `wandb` and specify a project/run name in the
Hydra overrides. Offline logging is not the same as an online run; verify the
run URL after startup.

## 7. What is still needed for exact reproduction

- A pinned environment lockfile tested on the target GPU stack.
- Audited data/model/retriever manifests and checksums.
- Canonical train and full-evaluation launchers for each benchmark.
- Successful focused pytest run and at least one end-to-end GPU smoke per
  benchmark.
- A saved resolved Hydra config and source revision for every reported result.
