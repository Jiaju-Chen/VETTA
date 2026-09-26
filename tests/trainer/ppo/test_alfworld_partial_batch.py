from agent_system.environments.env_package.alfworld import envs as alfworld_envs


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args):
        return self.fn(*args)


class FakeWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.reset = RemoteMethod(self._reset)
        self.step = RemoteMethod(self._step)
        self.getobs = RemoteMethod(lambda: f"image-{worker_id}")

    def _reset(self, game_index):
        info = {
            "admissible_commands": [["look"]],
            "extra.gamefile": [f"game-{game_index}"],
        }
        return [f"obs-{game_index}"], info

    def _step(self, action):
        info = {
            "admissible_commands": [[action]],
            "extra.gamefile": [f"game-{self.worker_id}"],
            "won": [False],
        }
        return [f"obs-{self.worker_id}"], [0.0], [False], info


def make_env(worker_count=20):
    env = alfworld_envs.AlfworldEnvs.__new__(alfworld_envs.AlfworldEnvs)
    env.num_processes = worker_count
    env.active_num_processes = worker_count
    env.workers = [FakeWorker(i) for i in range(worker_count)]
    env.prev_admissible_commands = [None] * worker_count
    env.is_train = False
    env.multi_modal = False
    return env


def test_partial_validation_batch_uses_only_active_workers(monkeypatch):
    monkeypatch.setattr(alfworld_envs.ray, "get", lambda values: values)
    env = make_env()

    observations, _, infos = env.reset(game_indices=list(range(14)))

    assert env.active_num_processes == 14
    assert len(observations) == 14
    assert len(infos) == 14
    assert len(env.get_admissible_commands) == 14

    next_observations, _, _, _, next_infos = env.step(["look"] * 14)

    assert len(next_observations) == 14
    assert len(next_infos) == 14


def test_full_batch_reactivates_all_workers(monkeypatch):
    monkeypatch.setattr(alfworld_envs.ray, "get", lambda values: values)
    env = make_env()
    env.reset(game_indices=list(range(14)))

    observations, _, _ = env.reset(game_indices=list(range(20)))

    assert env.active_num_processes == 20
    assert len(observations) == 20
    assert len(env.get_admissible_commands) == 20
