import os

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from verl.utils.model import update_model_config


SNAPSHOT = os.environ.get("QWEN_SNAPSHOT", "Qwen/Qwen2.5-1.5B-Instruct")


def main():
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    config = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True, attn_implementation="sdpa")
    update_model_config(
        config,
        override_config_kwargs={
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "num_hidden_layers": 2,
        },
    )
    config.num_labels = 1
    config.classifier_dropout = 0.0
    config.hidden_dropout = "0"

    model = AutoModelForTokenClassification.from_pretrained(
        SNAPSHOT,
        config=config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    assert len(model.model.layers) == 2
    assert model.score.out_features == 1

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"critic_class={model.__class__.__name__}")
    print(f"num_hidden_layers={len(model.model.layers)}")
    print(f"value_head_out_features={model.score.out_features}")
    print(f"total_parameters={total_parameters}")
    print(f"trainable_parameters={trainable_parameters}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    inputs = tokenizer(
        "Your task is to heat the apple and place it on the countertop.",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits

    assert logits.shape == (*inputs["input_ids"].shape, 1)
    assert torch.isfinite(logits).all()
    print(f"forward_shape={tuple(logits.shape)}")
    print("smoke_status=ok")


if __name__ == "__main__":
    main()
