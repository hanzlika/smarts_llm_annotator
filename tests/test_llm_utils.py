from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from smarts_llm_annotator import llm_utils


def _models_response(ids):
    return SimpleNamespace(data=[SimpleNamespace(id=i) for i in ids])


async def test_list_available_models_returns_ids():
    with patch.object(llm_utils.client.models, "list", new=AsyncMock(return_value=_models_response(["a", "b", "c"]))):
        models = await llm_utils.list_available_models()
    assert models == ["a", "b", "c"]


async def test_resolve_model_name_returns_explicit_value_without_querying_api():
    with patch.object(llm_utils, "list_available_models", new=AsyncMock(side_effect=AssertionError("should not be called"))):
        result = await llm_utils.resolve_model_name("gpt-4o-mini")
    assert result == "gpt-4o-mini"


async def test_resolve_model_name_defaults_to_first_when_not_interactive():
    with patch.object(llm_utils, "list_available_models", new=AsyncMock(return_value=["a", "b", "c"])):
        result = await llm_utils.resolve_model_name(None, interactive=False)
    assert result == "a"


async def test_resolve_model_name_defaults_to_first_outside_a_terminal():
    with patch.object(llm_utils, "list_available_models", new=AsyncMock(return_value=["a", "b", "c"])), \
         patch("sys.stdin.isatty", return_value=False):
        result = await llm_utils.resolve_model_name(None)
    assert result == "a"


async def test_resolve_model_name_prompts_and_returns_the_chosen_model():
    with patch.object(llm_utils, "list_available_models", new=AsyncMock(return_value=["a", "b", "c"])), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", return_value="2"):
        result = await llm_utils.resolve_model_name(None)
    assert result == "b"


async def test_resolve_model_name_blank_prompt_response_defaults_to_first():
    with patch.object(llm_utils, "list_available_models", new=AsyncMock(return_value=["a", "b", "c"])), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", return_value=""):
        result = await llm_utils.resolve_model_name(None)
    assert result == "a"


async def test_resolve_model_name_raises_if_no_models_available():
    with patch.object(llm_utils, "list_available_models", new=AsyncMock(return_value=[])):
        with pytest.raises(RuntimeError):
            await llm_utils.resolve_model_name(None, interactive=False)
