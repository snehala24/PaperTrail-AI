"""
Unit tests for Google Gemini LLM Provider (backend/app/llm/gemini.py).

Zero network calls: all tests use deterministic mocks.
All asynchronous operations are driven via run_sync to avoid Windows IOCP socket hangs.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from backend.app.core.config import get_gemini_api_key, settings
from backend.app.llm.base import LLMMessage, LLMProvider, LLMResponse
from backend.app.llm.gemini import GeminiProvider, _supports_temperature


def run_sync(coro):
    """Drive a coroutine to completion synchronously without requiring an OS socketpair."""
    if not hasattr(coro, "send"):
        return coro
    try:
        val = None
        while True:
            val = coro.send(val)
            if hasattr(val, "__await__"):
                sub = val.__await__()
                try:
                    sub_val = None
                    while True:
                        sub_val = sub.send(sub_val)
                except StopIteration as s:
                    val = s.value
    except StopIteration as e:
        return e.value


def make_mock_interaction(
    output_text="Test response from Gemini",
    model="gemini-3.8-flash",
    prompt_tokens=15,
    completion_tokens=25,
    total_tokens=40,
):
    """Helper to build a mock Interaction object returned by Interactions API."""
    usage = None
    if (
        prompt_tokens is not None
        or completion_tokens is not None
        or total_tokens is not None
    ):
        usage = SimpleNamespace(
            total_input_tokens=prompt_tokens,
            total_output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    return SimpleNamespace(
        output_text=output_text,
        model=model,
        usage=usage,
        steps=[],
    )


def make_mock_client(return_value=None, side_effect=None):
    """Helper to create a mocked google-genai Client."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.interactions = MagicMock()
    client.aio.interactions.create = AsyncMock(
        return_value=return_value,
        side_effect=side_effect,
    )
    client.aio.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Test 1: Successful generation returns expected LLMResponse
# ---------------------------------------------------------------------------
def test_successful_generation():
    mock_interaction = make_mock_interaction(
        output_text="PaperTrail is an AI research workspace.",
        model="gemini-3.8-flash",
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
    )
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    assert isinstance(provider, LLMProvider)

    messages = [
        LLMMessage(role="system", content="You are a helpful research assistant."),
        LLMMessage(role="user", content="What is PaperTrail?"),
    ]

    response = run_sync(provider.generate(messages, max_tokens=256))

    assert isinstance(response, LLMResponse)
    assert response.content == "PaperTrail is an AI research workspace."
    assert response.model == "gemini-3.8-flash"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 8
    assert response.total_tokens == 20

    mock_client.aio.interactions.create.assert_awaited_once()
    call_kwargs = mock_client.aio.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert call_kwargs["system_instruction"] == "You are a helpful research assistant."
    assert call_kwargs["input"] == "What is PaperTrail?"
    assert call_kwargs["generation_config"] == {"max_output_tokens": 256}


# ---------------------------------------------------------------------------
# Test 2: System and user messages are mapped correctly
# ---------------------------------------------------------------------------
def test_system_and_user_messages_mapped_correctly():
    mock_interaction = make_mock_interaction(output_text="Done")
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    # Test multiple system messages concatenated with \n\n
    messages = [
        LLMMessage(role="system", content="Part 1: You are an expert."),
        LLMMessage(role="system", content="Part 2: Cite evidence accurately."),
        LLMMessage(role="user", content="Summarize Section 3."),
    ]

    response = run_sync(provider.generate(messages))
    assert response.content == "Done"

    call_kwargs = mock_client.aio.interactions.create.call_args.kwargs
    assert (
        call_kwargs["system_instruction"]
        == "Part 1: You are an expert.\n\nPart 2: Cite evidence accurately."
    )
    assert call_kwargs["input"] == "Summarize Section 3."


# ---------------------------------------------------------------------------
# Test 3: Assistant messages handled correctly (multi-turn conversation)
# ---------------------------------------------------------------------------
def test_assistant_messages_handled_correctly():
    mock_interaction = make_mock_interaction(output_text="4")
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    messages = [
        LLMMessage(role="user", content="What is 1+1?"),
        LLMMessage(role="assistant", content="1+1 is 2."),
        LLMMessage(role="user", content="Multiply that by 2."),
    ]

    response = run_sync(provider.generate(messages))
    assert response.content == "4"

    call_kwargs = mock_client.aio.interactions.create.call_args.kwargs
    assert "system_instruction" not in call_kwargs

    expected_input = [
        {"type": "user_input", "content": [{"type": "text", "text": "What is 1+1?"}]},
        {
            "type": "model_output",
            "content": [{"type": "text", "text": "1+1 is 2."}],
        },
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Multiply that by 2."}],
        },
    ]
    assert call_kwargs["input"] == expected_input


# ---------------------------------------------------------------------------
# Test 4: Missing API key or model configuration fails clearly
# ---------------------------------------------------------------------------
def test_missing_api_key_raises():
    with patch.object(settings, "gemini_api_key", None):
        with pytest.raises(ValueError) as exc:
            GeminiProvider(api_key=None, model="gemini-3.8-flash")
        assert "GEMINI_API_KEY is not set" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            get_gemini_api_key()
        assert "GEMINI_API_KEY is not set" in str(exc.value)


def test_missing_model_raises():
    with patch.object(settings, "gemini_model", None):
        with pytest.raises(ValueError) as exc:
            GeminiProvider(api_key="valid-key", model=None)
        assert "GEMINI_MODEL is not set" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 5: Empty responses fail safely
# ---------------------------------------------------------------------------
def test_empty_response_text_none_raises():
    mock_interaction = make_mock_interaction(output_text=None)
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    messages = [LLMMessage(role="user", content="Hello")]
    with pytest.raises(RuntimeError) as exc:
        run_sync(provider.generate(messages))
    assert "empty response" in str(exc.value).lower()


def test_empty_response_whitespace_raises():
    mock_interaction = make_mock_interaction(output_text="   \n\t  ")
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    messages = [LLMMessage(role="user", content="Hello")]
    with pytest.raises(RuntimeError) as exc:
        run_sync(provider.generate(messages))
    assert "empty response" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Test 6: Retryable transient failures retry within configured bound
# ---------------------------------------------------------------------------
def test_retryable_429_eventually_succeeds():
    err_429 = genai_errors.ClientError(
        code=429,
        response_json={"error": {"message": "Resource has been exhausted"}},
    )
    mock_interaction = make_mock_interaction(output_text="Success after 429")

    # Fails twice with 429, succeeds on 3rd attempt
    mock_client = make_mock_client(
        side_effect=[err_429, err_429, mock_interaction]
    )

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        max_retries=3,
        initial_delay=0.001,
        max_delay=0.01,
    )

    messages = [LLMMessage(role="user", content="Test 429")]
    response = run_sync(provider.generate(messages))

    assert response.content == "Success after 429"
    assert mock_client.aio.interactions.create.await_count == 3


def test_retryable_503_eventually_succeeds():
    err_503 = genai_errors.ServerError(
        code=503,
        response_json={"error": {"message": "Service Unavailable"}},
    )
    mock_interaction = make_mock_interaction(output_text="Success after 503")

    mock_client = make_mock_client(side_effect=[err_503, mock_interaction])

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        max_retries=2,
        initial_delay=0.001,
    )

    messages = [LLMMessage(role="user", content="Test 503")]
    response = run_sync(provider.generate(messages))

    assert response.content == "Success after 503"
    assert mock_client.aio.interactions.create.await_count == 2


def test_retryable_failure_exhausts_retries():
    err_429 = genai_errors.ClientError(
        code=429,
        response_json={"error": {"message": "Rate limit exceeded"}},
    )
    mock_client = make_mock_client(side_effect=err_429)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        max_retries=2,
        initial_delay=0.001,
        max_delay=0.005,
    )

    messages = [LLMMessage(role="user", content="Test Exhaustion")]
    with pytest.raises(genai_errors.ClientError) as exc:
        run_sync(provider.generate(messages))
    assert exc.value.code == 429
    # 1 initial call + 2 retries = 3 calls
    assert mock_client.aio.interactions.create.await_count == 3


# ---------------------------------------------------------------------------
# Test 7: Permanent failures (401, 400) are NOT retried
# ---------------------------------------------------------------------------
def test_permanent_401_unauthorized_not_retried():
    err_401 = genai_errors.ClientError(
        code=401,
        response_json={"error": {"message": "API key not valid"}},
    )
    mock_client = make_mock_client(side_effect=err_401)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        max_retries=3,
        initial_delay=0.001,
    )

    messages = [LLMMessage(role="user", content="Test 401")]
    with pytest.raises(genai_errors.ClientError) as exc:
        run_sync(provider.generate(messages))
    assert exc.value.code == 401
    # Exactly 1 call, 0 retries
    assert mock_client.aio.interactions.create.await_count == 1


def test_permanent_400_bad_request_not_retried():
    err_400 = genai_errors.ClientError(
        code=400,
        response_json={"error": {"message": "Invalid argument"}},
    )
    mock_client = make_mock_client(side_effect=err_400)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        max_retries=3,
        initial_delay=0.001,
    )

    messages = [LLMMessage(role="user", content="Test 400")]
    with pytest.raises(genai_errors.ClientError) as exc:
        run_sync(provider.generate(messages))
    assert exc.value.code == 400
    assert mock_client.aio.interactions.create.await_count == 1


# ---------------------------------------------------------------------------
# Test 8: Timeouts handled safely
# ---------------------------------------------------------------------------
def test_timeout_retries_and_exhausts():
    mock_client = make_mock_client(side_effect=asyncio.TimeoutError())

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        timeout=0.05,
        max_retries=2,
        initial_delay=0.001,
    )

    messages = [LLMMessage(role="user", content="Test Timeout")]
    with pytest.raises(asyncio.TimeoutError):
        run_sync(provider.generate(messages))
    # 1 initial + 2 retries = 3 calls
    assert mock_client.aio.interactions.create.await_count == 3
    assert mock_client.aio.interactions.create.call_args.kwargs["timeout"] == 0.05


def test_timeout_passed_to_sdk_interactions_create():
    mock_interaction = make_mock_interaction(output_text="Response")
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        timeout=45.0,
    )

    run_sync(provider.generate([LLMMessage(role="user", content="Hello")]))
    mock_client.aio.interactions.create.assert_awaited_once()
    assert mock_client.aio.interactions.create.call_args.kwargs["timeout"] == 45.0


def test_cancellation_not_retried():
    mock_client = make_mock_client(side_effect=asyncio.CancelledError())

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
        max_retries=3,
        initial_delay=0.001,
    )

    messages = [LLMMessage(role="user", content="Test Cancellation")]
    with pytest.raises(asyncio.CancelledError):
        run_sync(provider.generate(messages))
    # Never retry cancellation
    assert mock_client.aio.interactions.create.await_count == 1


# ---------------------------------------------------------------------------
# Test 9: Invalid inputs rejected
# ---------------------------------------------------------------------------
def test_empty_messages_list_rejected():
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=make_mock_client(),
    )
    with pytest.raises(ValueError) as exc:
        run_sync(provider.generate([]))
    assert "cannot be empty" in str(exc.value)


def test_only_system_messages_rejected():
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=make_mock_client(),
    )
    with pytest.raises(ValueError) as exc:
        run_sync(provider.generate([
            LLMMessage(role="system", content="System instruction only")
        ]))
    assert "at least one user or assistant message" in str(exc.value)


def test_unsupported_role_rejected():
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=make_mock_client(),
    )
    with pytest.raises(ValueError) as exc:
        run_sync(provider.generate([
            LLMMessage(role="developer", content="Developer prompt")
        ]))
    assert "Unsupported message role" in str(exc.value)


def test_negative_max_tokens_rejected():
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=make_mock_client(),
    )
    with pytest.raises(ValueError) as exc:
        run_sync(provider.generate(
            [LLMMessage(role="user", content="Hi")], max_tokens=-10
        ))
    assert "max_tokens must be positive" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        run_sync(provider.generate(
            [LLMMessage(role="user", content="Hi")], max_tokens=0
        ))
    assert "max_tokens must be positive" in str(exc.value)


def test_negative_temperature_rejected():
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=make_mock_client(),
    )
    with pytest.raises(ValueError) as exc:
        run_sync(provider.generate(
            [LLMMessage(role="user", content="Hi")], temperature=-0.5
        ))
    assert "temperature must be non-negative" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 10: Token usage handled correctly when present and when absent
# ---------------------------------------------------------------------------
def test_token_usage_present():
    mock_interaction = make_mock_interaction(
        output_text="Tokens test",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    response = run_sync(
        provider.generate([LLMMessage(role="user", content="Hi")])
    )
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 50
    assert response.total_tokens == 150


def test_token_usage_absent():
    mock_interaction = make_mock_interaction(
        output_text="Tokens absent test",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )
    mock_interaction.usage = None  # Completely absent usage
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    response = run_sync(
        provider.generate([LLMMessage(role="user", content="Hi")])
    )
    assert response.content == "Tokens absent test"
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.total_tokens is None


# ---------------------------------------------------------------------------
# Test 11: Logs do not expose credentials or full prompts
# ---------------------------------------------------------------------------
def test_logs_do_not_expose_credentials_or_prompts(caplog):
    secret_key = "AIzaSySuperSecretKeyNotForLogs12345"
    confidential_prompt = "TOP_SECRET_PROMPT_CONTAINING_PRIVATE_DATA_DO_NOT_LOG"
    sensitive_response = "SENSITIVE_ANSWER_NOT_TO_BE_LOGGED_EITHER"

    mock_interaction = make_mock_interaction(output_text=sensitive_response)
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key=secret_key,
        model="gemini-3.8-flash",
        client=mock_client,
    )

    with caplog.at_level(logging.DEBUG):
        run_sync(provider.generate([
            LLMMessage(role="user", content=confidential_prompt)
        ]))

    all_logs = " ".join(record.message for record in caplog.records)
    assert secret_key not in all_logs
    assert confidential_prompt not in all_logs
    assert sensitive_response not in all_logs
    assert "gemini-3.8-flash" in all_logs
    assert "prompt_tokens:" in all_logs


# ---------------------------------------------------------------------------
# Test 12: Client cleanup works as expected
# ---------------------------------------------------------------------------
def test_client_cleanup_aclose():
    mock_client = make_mock_client()

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    run_sync(provider.aclose())
    mock_client.aio.aclose.assert_awaited_once()


def test_client_cleanup_async_context_manager():
    mock_client = make_mock_client()

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    async def with_manager():
        async with provider as p:
            assert p is provider

    run_sync(with_manager())
    mock_client.aio.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 13: Temperature deprecation support check
# ---------------------------------------------------------------------------
def test_temperature_support_helper():
    assert not _supports_temperature("gemini-3.8-flash")
    assert not _supports_temperature("gemini-3.7-flash")
    assert not _supports_temperature("gemini-3.6-flash")
    assert not _supports_temperature("gemini-4-argon")
    assert _supports_temperature("gemini-2.0-flash")
    assert _supports_temperature("gemini-1.5-pro")


def test_temperature_passed_for_supported_models():
    mock_interaction = make_mock_interaction(output_text="Answer")
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-2.0-flash",
        client=mock_client,
    )

    run_sync(
        provider.generate(
            [LLMMessage(role="user", content="Hi")], temperature=0.7
        )
    )
    call_kwargs = mock_client.aio.interactions.create.call_args.kwargs
    assert call_kwargs["generation_config"]["temperature"] == 0.7


def test_temperature_omitted_for_gemini_38_flash():
    mock_interaction = make_mock_interaction(output_text="Answer")
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    run_sync(
        provider.generate(
            [LLMMessage(role="user", content="Hi")], temperature=0.7
        )
    )
    call_kwargs = mock_client.aio.interactions.create.call_args.kwargs
    # Temperature omitted to avoid deprecation warning for Gemini 3.8 Flash
    assert "generation_config" not in call_kwargs or (
        "temperature" not in call_kwargs.get("generation_config", {})
    )


def test_failed_interaction_with_errors_raises():
    mock_interaction = make_mock_interaction(output_text=None)
    mock_interaction.errors = [
        SimpleNamespace(code="RESOURCE_EXHAUSTED", message="Quota exceeded for model")
    ]
    mock_client = make_mock_client(return_value=mock_interaction)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        client=mock_client,
    )

    with pytest.raises(RuntimeError) as exc:
        run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))
    assert "RESOURCE_EXHAUSTED" in str(exc.value)
    assert "Quota exceeded" in str(exc.value)


def test_sdk_internal_retries_disabled_on_default_client():
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.8-flash",
    )
    aio = getattr(provider._client, "aio", None)
    cfg = getattr(aio.interactions, "sdk_configuration", None)
    assert cfg is not None
    assert getattr(cfg.retry_config, "strategy", None) == "none"
    assert getattr(cfg.retry_config, "max_retries", None) == 0
