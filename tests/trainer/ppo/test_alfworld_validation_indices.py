import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import extract_alfworld_game_indices


def make_batch(non_tensor_batch):
    return DataProto.from_single_dict(
        {
            "input_ids": torch.zeros((len(next(iter(non_tensor_batch.values()))), 1), dtype=torch.long),
            **non_tensor_batch,
        }
    )


def test_prefers_collated_tensor_indexes():
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.zeros((20, 1), dtype=torch.long),
            "index": torch.arange(20),
            "extra_info": np.array([{"index": 99}] * 20, dtype=object),
        }
    )

    np.testing.assert_array_equal(extract_alfworld_game_indices(batch), np.arange(20))


def test_supports_non_tensor_top_level_indexes():
    batch = make_batch({"index": np.arange(20, dtype=object)})

    np.testing.assert_array_equal(extract_alfworld_game_indices(batch), np.arange(20))


def test_supports_legacy_extra_info_indexes():
    batch = make_batch({"extra_info": np.array([{"index": 7}, {"index": 8}], dtype=object)})

    np.testing.assert_array_equal(extract_alfworld_game_indices(batch), np.array([7, 8]))


def test_rejects_missing_indexes():
    batch = make_batch({"raw_prompt": np.array(["a", "b"], dtype=object)})

    try:
        extract_alfworld_game_indices(batch)
    except KeyError as error:
        assert "index or extra_info.index" in str(error)
    else:
        raise AssertionError("Missing ALFWorld validation indexes must fail")


if __name__ == "__main__":
    test_prefers_collated_tensor_indexes()
    test_supports_non_tensor_top_level_indexes()
    test_supports_legacy_extra_info_indexes()
    test_rejects_missing_indexes()
    print("PASS ALFWorld validation index tests")
