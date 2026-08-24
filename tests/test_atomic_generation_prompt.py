
from src.generation.generator import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)


def test_system_prompt_requires_atomic_claims():
    prompt = SYSTEM_PROMPT.lower()

    assert "one factual claim per sentence" in prompt


def test_system_prompt_requires_separate_list_items():
    prompt = SYSTEM_PROMPT.lower()

    assert "each list item" in prompt
    assert "separate sentence" in prompt


def test_user_prompt_reinforces_atomic_claims():
    prompt = USER_TEMPLATE.lower()

    assert "one factual claim per sentence" in prompt



def test_prompt_forbids_inferred_rankings():
    prompt = SYSTEM_PROMPT.lower()

    assert "do not infer rankings" in prompt
    assert "list order" in prompt


def test_prompt_preserves_source_terminology():
    prompt = SYSTEM_PROMPT.lower()

    assert "do not expand abbreviations" in prompt
    assert "explicitly defines" in prompt


def test_prompt_requires_only_directly_supported_facts():
    prompt = SYSTEM_PROMPT.lower()

    assert "directly supported" in prompt
