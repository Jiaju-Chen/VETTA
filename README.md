# VETTA

VETTA studies critic-based credit assignment for long-horizon language-agent PPO.
This repository is a clean research-code snapshot for the method and its
algorithmic controls. It does not contain model weights, benchmark data, run
logs, or credentials.

## Method

The main critic reuses a shallow Transformer backbone and predicts token-level
and turn-boundary values with separate heads. The actor combines turn-level
advantage with a centered token residual:

```text
A[t,j] = A_turn[t] + alpha * (A_token[t,j] - mean_j(A_token[t,j]))
```

The centered residual lets the turn signal carry cross-turn progress while the
token signal distinguishes decisions within one response. The implementation
also includes single-head/shared-critic ablations and a Turn-PPO baseline used
for controlled comparisons.

## Code lineage

The rollout, environment, PPO, Ray, FSDP, and vLLM infrastructure is derived
from the public GiGPO/`verl-agent` and veRL codebases. Their licenses and notices
are retained in this snapshot. VETTA changes the critic/value path and advantage
composition; it is not a reimplementation of GiGPO's group-in-group objective.

## Benchmarks and assets

The code contains integrations for ALFWorld, WebShop, and SearchQA. Their data,
retrieval indexes/services, model checkpoints, and runtime secrets must be
obtained and configured separately under the corresponding upstream licenses.
Runtime paths, service endpoints, and credentials are supplied through local
configuration rather than embedded in the source.

Start with [Getting Started](docs/GETTING_STARTED.md) for environment setup,
smoke checks, baseline commands, VETTA commands, and the currently supported
best-observed settings. [Reproducibility](docs/REPRODUCIBILITY.md) describes
the remaining artifact and testing limits.

## License

The repository retains the upstream Apache-2.0 license and attribution notice.
See `LICENSE` and `Notice.txt`.
