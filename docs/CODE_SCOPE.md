# Code Scope

This snapshot contains the shared-critic VETTA implementation, its focused
algorithmic controls, relevant tests, and the inherited training/environment
framework.

Included VETTA changes cover:

- token/turn value prediction and the unified shared-backbone critic;
- residual/direct and single-head value/advantage paths;
- Turn-PPO GAE and response-level ratio code for baseline comparison;
- explicit validation aggregation and action-format diagnostic helpers;
- checkpoint save/resume support for the critic state.

Excluded from this snapshot are run-specific launchers with machine paths,
experiment logs and evidence exports, model/data artifacts, credentials,
unfinished visual-Sokoban research code, and version-control history.
Hardware-ready benchmark recipes will be added only after their
resolved configurations and data manifests are audited.
