from verl.trainer.ppo.action_format_metrics import summarize_action_format_metrics


def test_search_action_format_metrics_distinguish_executable_from_strict_format():
    metrics = summarize_action_format_metrics(
        [
            "<search>alpha</search>",
            "<search>beta</search><answer>beta</answer>",
            "<search>first</search><search>second</search>",
            "plain text",
        ]
    )

    assert metrics["episode/action_format/executable_block_ratio"] == 0.75
    assert metrics["episode/action_format/missing_executable_block_ratio"] == 0.25
    assert metrics["episode/action_format/search_block_ratio"] == 0.75
    assert metrics["episode/action_format/answer_block_ratio"] == 0.25
    assert metrics["episode/action_format/mixed_search_answer_ratio"] == 0.25
    assert metrics["episode/action_format/repeated_control_tag_ratio"] == 0.25


def test_webshop_action_format_metrics_expose_penalty_reasons():
    metrics = summarize_action_format_metrics(
        [
            "<think>reason</think><action>search[item]</action>",
            "<action>click[item]</action>",
            "<think>reason</think><action>buy now</action>中文",
        ]
    )

    assert metrics["episode/action_format/executable_block_ratio"] == 1.0
    assert metrics["episode/action_format/webshop_action_block_ratio"] == 1.0
    assert metrics["episode/action_format/think_block_ratio"] == 2 / 3
    assert metrics["episode/action_format/chinese_text_ratio"] == 1 / 3
