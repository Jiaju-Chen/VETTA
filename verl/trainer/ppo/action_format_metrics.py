"""Behavior-preserving diagnostics for structured environment actions."""

import re
from collections.abc import Iterable


_SEARCH_BLOCK = re.compile(r"<search>.*?</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)
_ACTION_BLOCK = re.compile(r"<action>.*?</action>", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_SEARCH_TAG = re.compile(r"<search>", re.IGNORECASE)
_ANSWER_TAG = re.compile(r"<answer>", re.IGNORECASE)
_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)
_CHINESE_TEXT = re.compile(r"[\u4e00-\u9fff]")


def _iter_action_texts(values):
    if isinstance(values, str):
        yield values
        return
    if hasattr(values, "flat"):
        values = values.flat
    if isinstance(values, Iterable):
        for value in values:
            yield from _iter_action_texts(value)


def summarize_action_format_metrics(action_texts):
    """Summarize parser-relevant output patterns without changing projection."""
    texts = list(_iter_action_texts(action_texts))
    if not texts:
        return {}

    counts = {
        "executable_block": 0,
        "missing_executable_block": 0,
        "search_block": 0,
        "answer_block": 0,
        "webshop_action_block": 0,
        "mixed_search_answer": 0,
        "repeated_control_tag": 0,
        "think_block": 0,
        "chinese_text": 0,
    }

    for text in texts:
        has_search = bool(_SEARCH_BLOCK.search(text))
        has_answer = bool(_ANSWER_BLOCK.search(text))
        has_action = bool(_ACTION_BLOCK.search(text))
        has_executable = has_search or has_answer or has_action

        counts["executable_block"] += has_executable
        counts["missing_executable_block"] += not has_executable
        counts["search_block"] += has_search
        counts["answer_block"] += has_answer
        counts["webshop_action_block"] += has_action
        counts["mixed_search_answer"] += has_search and has_answer
        counts["repeated_control_tag"] += (
            len(_SEARCH_TAG.findall(text)) > 1
            or len(_ANSWER_TAG.findall(text)) > 1
            or len(_ACTION_TAG.findall(text)) > 1
        )
        counts["think_block"] += bool(_THINK_BLOCK.search(text))
        counts["chinese_text"] += bool(_CHINESE_TEXT.search(text))

    denominator = float(len(texts))
    return {
        f"episode/action_format/{name}_ratio": count / denominator
        for name, count in counts.items()
    }
