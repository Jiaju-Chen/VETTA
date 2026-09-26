import pathlib
import unittest


class CriticOverrideRegressionTest(unittest.TestCase):
    def test_critic_applies_override_before_model_construction(self):
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        source = (repo_root / "verl/workers/fsdp_workers.py").read_text()

        critic_start = source.index("class CriticWorker")
        critic_source = source[critic_start:]
        config_load = critic_source.index("critic_model_config = AutoConfig.from_pretrained")
        override_call = "update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)"
        self.assertIn(override_call, critic_source)
        override_apply = critic_source.index(override_call)
        model_load = critic_source.index("critic_module = AutoModelForTokenClassification.from_pretrained")

        self.assertLess(config_load, override_apply)
        self.assertLess(override_apply, model_load)


if __name__ == "__main__":
    unittest.main()
