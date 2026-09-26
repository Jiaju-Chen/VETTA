# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import numpy as np
import os
import ray

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seeds, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        self.envs = []
        for seed in seeds:
            worker_env_kwargs = dict(env_kwargs)
            worker_env_kwargs['seed'] = seed
            self.envs.append(gym.make('WebAgentTextEnv-v0', **worker_env_kwargs))
    
    def step(self, actions):
        """Execute one action for every environment in this worker shard."""
        if len(actions) != len(self.envs):
            raise ValueError(f'Expected {len(self.envs)} actions, got {len(actions)}')

        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            info = dict(info or {})
            info['available_actions'] = env.get_available_actions()
            info['task_score'] = reward

            # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
            if done and reward == 1.0:
                info['won'] = True
                reward = 10.0
            else:
                info['won'] = False
                reward = 0
            results.append((obs, reward, done, info))
        return results
    
    def reset(self, indices):
        """Reset every environment in this worker shard."""
        if len(indices) != len(self.envs):
            raise ValueError(f'Expected {len(self.envs)} indices, got {len(indices)}')

        results = []
        for env, idx in zip(self.envs, indices):
            obs, info = env.reset(session=idx)
            info = dict(info or {})
            info['goal_index'] = int(idx)
            info['available_actions'] = env.get_available_actions()
            info['won'] = False
            results.append((obs, info))
        return results
    
    def render(self, mode_for_render, local_index=None):
        """Render the environment"""
        if local_index is not None:
            return self.envs[local_index].render(mode=mode_for_render)
        return [env.render(mode=mode_for_render) for env in self.envs]
    
    def get_available_actions(self):
        """Get available actions"""
        return [env.get_available_actions() for env in self.envs]
    
    def get_goals(self):
        """Get environment goals"""
        return self.envs[0].server.goals

    def ready(self):
        """Confirm that the embedded Python and Java environment is initialized."""
        return True
    
    def close(self):
        """Close the environment"""
        for env in self.envs:
            env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        worker_env_batch_size = max(
            int(os.environ.get('WEBSHOP_ENVS_PER_WORKER', '8')),
            1,
        )
        worker_resources = dict(resources_per_worker)
        if 'num_cpus' in worker_resources:
            worker_resources['num_cpus'] *= worker_env_batch_size
        if 'num_gpus' in worker_resources:
            worker_resources['num_gpus'] *= worker_env_batch_size
        env_worker = ray.remote(**worker_resources)(WebshopWorker)
        self._workers = []
        self._worker_slices = []
        init_batch_size = max(int(os.environ.get('WEBSHOP_ENV_INIT_BATCH_SIZE', '4')), 1)
        pending_ready = []
        for start in range(0, self.num_processes, worker_env_batch_size):
            stop = min(start + worker_env_batch_size, self.num_processes)
            seeds = [seed + (i // self.group_n) for i in range(start, stop)]
            worker = env_worker.remote(seeds, self._env_kwargs)
            self._workers.append(worker)
            self._worker_slices.append(slice(start, stop))
            pending_ready.append(worker.ready.remote())
            if len(pending_ready) == init_batch_size or stop == self.num_processes:
                ray.get(pending_ready)
                pending_ready = []

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = [
            worker.step.remote(actions[worker_slice])
            for worker, worker_slice in zip(self._workers, self._worker_slices)
        ]

        # Collect results
        results = [item for shard in ray.get(futures) for item in shard]
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self, goal_indices=None):
        if goal_indices is None:
            idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        else:
            idx = np.asarray(goal_indices, dtype=np.int64).reshape(-1)
            if len(idx) != self.env_num:
                raise ValueError(
                    f'Expected {self.env_num} goal indices, got {len(idx)}',
                )
            invalid = [int(i) for i in idx if i not in self.goal_idxs]
            if invalid:
                raise ValueError(f'Goal indices outside this split: {invalid}')
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = [
            worker.reset.remote(idx[worker_slice])
            for worker, worker_slice in zip(self._workers, self._worker_slices)
        ]

        # Collect results
        results = [item for shard in ray.get(futures) for item in shard]
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            for worker, worker_slice in zip(self._workers, self._worker_slices):
                if worker_slice.start <= env_idx < worker_slice.stop:
                    local_index = env_idx - worker_slice.start
                    future = worker.render.remote(mode, local_index)
                    return ray.get(future)
            raise IndexError(f'Environment index out of range: {env_idx}')

        futures = [worker.render.remote(mode) for worker in self._workers]
        return [item for shard in ray.get(futures) for item in shard]

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
