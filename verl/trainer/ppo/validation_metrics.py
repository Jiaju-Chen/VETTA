"""Validation metrics that operate on one outcome per environment episode."""

from collections import OrderedDict

import numpy as np


def unique_episode_outcomes(data_sources, traj_uids, episode_successes):
    """Deduplicate expanded turn rows while preserving trajectory order.

    Multi-turn validation expands one environment episode into several rows.  The
    task label and terminal outcome must agree across every row of a trajectory.
    """
    data_sources = np.asarray(data_sources).reshape(-1)
    traj_uids = np.asarray(traj_uids).reshape(-1)
    episode_successes = np.asarray(episode_successes).reshape(-1)

    if not (len(data_sources) == len(traj_uids) == len(episode_successes)):
        raise ValueError(
            "data_sources, traj_uids, and episode_successes must have equal lengths"
        )

    episodes = OrderedDict()
    for source, traj_uid, success in zip(data_sources, traj_uids, episode_successes):
        uid = traj_uid.item() if isinstance(traj_uid, np.generic) else traj_uid
        source = source.item() if isinstance(source, np.generic) else source
        success = success.item() if isinstance(success, np.generic) else success
        outcome = (str(source), float(success))

        if uid in episodes:
            if episodes[uid] != outcome:
                raise ValueError(
                    f"Inconsistent validation rows for trajectory {uid!r}: "
                    f"{episodes[uid]!r} != {outcome!r}"
                )
            continue
        episodes[uid] = outcome

    unique_sources = np.asarray([outcome[0] for outcome in episodes.values()], dtype=object)
    unique_successes = np.asarray(
        [outcome[1] for outcome in episodes.values()], dtype=np.float32
    )
    return unique_sources, unique_successes


def summarize_episode_successes(data_sources, episode_successes):
    """Compute exact overall and per-task success rates from unique episodes."""
    data_sources = np.asarray(data_sources).reshape(-1)
    episode_successes = np.asarray(episode_successes, dtype=np.float32).reshape(-1)
    if len(data_sources) != len(episode_successes):
        raise ValueError("data_sources and episode_successes must have equal lengths")
    if not len(episode_successes):
        raise ValueError("cannot summarize an empty validation set")

    metrics = {
        "success_rate": float(episode_successes.mean()),
        "evaluated_cases": int(len(episode_successes)),
    }
    for source in dict.fromkeys(str(source) for source in data_sources):
        mask = np.asarray([str(item) == source for item in data_sources])
        metrics[f"{source}_success_rate"] = float(episode_successes[mask].mean())
        metrics[f"{source}_evaluated_cases"] = int(mask.sum())
    return metrics
