# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import ast
import sys
from pathlib import Path

from verl.utils.tracking import Tracking


class _FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def log(self, **kwargs):
        self.log_calls.append(kwargs)

    def finish(self, **kwargs):
        self.finish_calls.append(kwargs)


def test_finish_flushes_wandb_once(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracking = Tracking(
        project_name="test-project",
        experiment_name="test-run",
        default_backend="wandb",
    )
    tracking.log({"training/global_step": 150}, step=150)
    tracking.finish()
    tracking.finish()
    tracking.__del__()

    assert fake_wandb.log_calls == [
        {"data": {"training/global_step": 150}, "step": 150}
    ]
    assert fake_wandb.finish_calls == [{"exit_code": 0}]


def test_ppo_fit_explicitly_finishes_tracking_on_normal_returns():
    repo_root = Path(__file__).resolve().parents[3]
    trainer_path = repo_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    module = ast.parse(trainer_path.read_text())
    fit = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "fit"
    )

    finish_calls = [
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
        and node.func.attr == "finish"
    ]

    # One call handles val-only mode and one handles the completed training loop.
    assert len(finish_calls) >= 2
