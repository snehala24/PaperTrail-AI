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


# ---------------------------------------------------------------------------
# Test: Transient Error Classification and Bounded Retries
# ---------------------------------------------------------------------------


def test_is_retryable_error_gaos_compat_internal_server_error_503():
    """Observed SDK 503 InternalServerError is classified as retryable and retries within bounds."""
    import httpx
    from google.genai._gaos.lib import compat_errors

    req = httpx.Request("POST", "http://test")
    resp_503 = httpx.Response(503, request=req)
    err_503 = compat_errors.InternalServerError(
        "503 Service Unavailable: gemini-3.6-flash is currently experiencing high demand",
        response=resp_503,
        body={},
    )

    mock_client = make_mock_client()
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.6-flash",
        client=mock_client,
        max_retries=2,
        initial_delay=0.001,
        max_delay=0.005,
    )

    is_retryable, reason = provider._is_retryable_error(err_503)
    assert is_retryable is True
    assert "503" in reason

    # Test full generate loop: fails once with 503, succeeds on 2nd attempt
    mock_interaction = make_mock_interaction(output_text="Success after 503 retry")
    mock_client.aio.interactions.create = AsyncMock(
        side_effect=[err_503, mock_interaction]
    )

    response = run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))
    assert response.content == "Success after 503 retry"
    assert mock_client.aio.interactions.create.await_count == 2


def test_is_retryable_error_server_errors_500_502_504():
    """Transient server errors 500, 502, and 504 are retryable."""
    import httpx
    from google.genai._gaos.lib import compat_errors

    mock_client = make_mock_client()
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.6-flash",
        client=mock_client,
    )

    req = httpx.Request("POST", "http://test")
    for code in (500, 502, 504):
        resp = httpx.Response(code, request=req)
        err = compat_errors.APIError.generate(
            status_code=code,
            body={"error": {"code": code, "message": "Server error"}},
            message=None,
            response=resp,
        )
        is_retryable, reason = provider._is_retryable_error(err)
        assert is_retryable is True, f"Expected {code} to be retryable"
        assert str(code) in reason


def test_is_retryable_error_permanent_4xx_not_retryable():
    """Client errors (400, 401, 403, 404, 422) are non-transient and must never be retried."""
    import httpx
    from google.genai._gaos.lib import compat_errors

    mock_client = make_mock_client()
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.6-flash",
        client=mock_client,
        max_retries=2,
    )

    req = httpx.Request("POST", "http://test")
    client_error_classes = [
        (400, compat_errors.BadRequestError),
        (401, compat_errors.AuthenticationError),
        (403, compat_errors.PermissionDeniedError),
        (404, compat_errors.NotFoundError),
        (422, compat_errors.UnprocessableEntityError),
    ]

    for code, cls in client_error_classes:
        resp = httpx.Response(code, request=req)
        err = cls(f"HTTP {code} error", response=resp, body={})
        is_retryable, reason = provider._is_retryable_error(err)
        assert is_retryable is False, f"HTTP {code} should not be retryable"
        assert "Permanent client error" in reason
        assert str(code) in reason


def test_is_retryable_error_arbitrary_exceptions_not_retryable():
    """Arbitrary exceptions with status_code attribute or generic exceptions are not retried."""
    mock_client = make_mock_client()
    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.6-flash",
        client=mock_client,
    )

    class CustomErrorWithStatusCode(Exception):
        status_code = 503

    err = CustomErrorWithStatusCode("Not an SDK error")
    is_retryable, reason = provider._is_retryable_error(err)
    assert is_retryable is False
    assert "Non-retryable exception" in reason

    for generic in (ValueError("invalid"), TypeError("type error"), RuntimeError("runtime")):
        is_retryable, reason = provider._is_retryable_error(generic)
        assert is_retryable is False
        assert "Non-retryable exception" in reason


def test_retry_count_bounded_at_configured_maximum():
    """Provider never exceeds configured max_retries on persistent 503 errors."""
    import httpx
    from google.genai._gaos.lib import compat_errors

    req = httpx.Request("POST", "http://test")
    resp_503 = httpx.Response(503, request=req)
    err_503 = compat_errors.InternalServerError(
        "503 Service Unavailable",
        response=resp_503,
        body={},
    )

    mock_client = make_mock_client()
    mock_client.aio.interactions.create = AsyncMock(side_effect=err_503)

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.6-flash",
        client=mock_client,
        max_retries=2,
        initial_delay=0.001,
        max_delay=0.005,
    )

    with pytest.raises(compat_errors.InternalServerError):
        run_sync(provider.generate([LLMMessage(role="user", content="Test persistent 503")]))

    # 1 initial attempt + 2 retries = 3 total attempts
    assert mock_client.aio.interactions.create.await_count == 3


def test_timeout_enforced_across_retry_attempts():
    """Per-attempt timeout argument is consistently passed to interactions.create on all attempts."""
    import httpx
    from google.genai._gaos.lib import compat_errors

    req = httpx.Request("POST", "http://test")
    resp_503 = httpx.Response(503, request=req)
    err_503 = compat_errors.InternalServerError("503", response=resp_503, body={})
    mock_interaction = make_mock_interaction(output_text="Success on attempt 2")

    mock_client = make_mock_client()
    mock_client.aio.interactions.create = AsyncMock(
        side_effect=[err_503, mock_interaction]
    )

    provider = GeminiProvider(
        api_key="test-api-key",
        model="gemini-3.6-flash",
        client=mock_client,
        timeout=15.0,
        max_retries=2,
        initial_delay=0.001,
        max_delay=0.005,
    )

    response = run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))
    assert response.content == "Success on attempt 2"
    assert mock_client.aio.interactions.create.await_count == 2

    # Check timeout kwarg was passed in all calls
    for call_args in mock_client.aio.interactions.create.call_args_list:
        assert call_args.kwargs.get("timeout") == 15.0


def test_interaction_steps_only_extracts_model_output():
    """Fallback step extraction never leaks user_input or thought steps into response content."""
    user_step = SimpleNamespace(
        type="user_input",
        content=[SimpleNamespace(text="User prompt that must never leak")],
    )
    thought_step = SimpleNamespace(
        type="thought",
        content=[SimpleNamespace(text="Internal model thought reasoning")],
    )
    model_step = SimpleNamespace(
        type="model_output",
        content=[SimpleNamespace(text='{"answer": "Verified output", "claims": []}')],
    )
    mock_interaction = SimpleNamespace(
        output_text="",
        model="gemini-3.6-flash",
        usage=None,
        steps=[user_step, thought_step, model_step],
    )
    mock_client = make_mock_client(return_value=mock_interaction)
    provider = GeminiProvider(api_key="test-api-key", model="gemini-3.6-flash", client=mock_client)

    response = run_sync(provider.generate([LLMMessage(role="user", content="Question")]))
    assert response.content == '{"answer": "Verified output", "claims": []}'
    assert "User prompt" not in response.content
    assert "Internal model thought" not in response.content


def test_interaction_steps_without_model_output_raises_empty_response():
    """If interaction only contains user_input or thought steps and no model_output, raises RuntimeError."""
    user_step = SimpleNamespace(
        type="user_input",
        content=[SimpleNamespace(text="User prompt")],
    )
    thought_step = SimpleNamespace(
        type="thought",
        content=[SimpleNamespace(text="Internal model thought")],
    )
    mock_interaction = SimpleNamespace(
        output_text=None,
        model="gemini-3.6-flash",
        usage=None,
        steps=[user_step, thought_step],
    )
    mock_client = make_mock_client(return_value=mock_interaction)
    provider = GeminiProvider(api_key="test-api-key", model="gemini-3.6-flash", client=mock_client)

    with pytest.raises(RuntimeError) as exc_info:
        run_sync(provider.generate([LLMMessage(role="user", content="Question")]))
    assert "empty response" in str(exc_info.value).lower()


def test_build_response_completed_status_sets_stop_finish_reason():
    """interaction.status == 'completed' sets finish_reason='stop'."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "ok"}',
        status="completed",
        model="gemini-3.6-flash",
        usage=SimpleNamespace(
            total_input_tokens=50,
            total_output_tokens=100,
            total_tokens=150,
            total_thought_tokens=20,
        ),
        steps=[],
    )
    provider = GeminiProvider(api_key="test-api-key", client=make_mock_client(return_value=mock_interaction))
    resp = run_sync(provider.generate([LLMMessage(role="user", content="Hi")], max_tokens=1024))
    assert resp.status == "completed"
    assert resp.finish_reason == "stop"
    assert resp.thought_tokens == 20


def test_build_response_incomplete_status():
    """interaction.status == 'incomplete' sets finish_reason='incomplete'."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "partial',
        status="incomplete",
        model="gemini-3.6-flash",
        usage=SimpleNamespace(total_input_tokens=50, total_output_tokens=100, total_tokens=150),
        steps=[],
    )
    provider = GeminiProvider(api_key="test-api-key", client=make_mock_client(return_value=mock_interaction))
    resp = run_sync(provider.generate([LLMMessage(role="user", content="Hi")], max_tokens=1024))
    assert resp.status == "incomplete"
    assert resp.finish_reason == "incomplete"


def test_build_response_budget_exceeded_status():
    """interaction.status == 'budget_exceeded' sets finish_reason='max_tokens'."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "partial',
        status="budget_exceeded",
        model="gemini-3.6-flash",
        usage=SimpleNamespace(total_input_tokens=50, total_output_tokens=500, total_tokens=550),
        steps=[],
    )
    provider = GeminiProvider(api_key="test-api-key", client=make_mock_client(return_value=mock_interaction))
    resp = run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))
    assert resp.status == "budget_exceeded"
    assert resp.finish_reason == "max_tokens"


def test_build_response_tokens_reach_max_tokens_sets_max_tokens():
    """When completion_tokens reaches configured max_tokens, finish_reason='max_tokens'."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "hit limit',
        status="completed",
        model="gemini-3.6-flash",
        usage=SimpleNamespace(total_input_tokens=50, total_output_tokens=1024, total_tokens=1074),
        steps=[],
    )
    provider = GeminiProvider(api_key="test-api-key", client=make_mock_client(return_value=mock_interaction))
    resp = run_sync(provider.generate([LLMMessage(role="user", content="Hi")], max_tokens=1024))
    assert resp.finish_reason == "max_tokens"


def test_build_response_missing_status_and_usage_handled_gracefully():
    """Interaction without status or usage fields defaults to None without crashing."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "ok"}',
        model="gemini-3.6-flash",
        steps=[],
    )
    provider = GeminiProvider(api_key="test-api-key", client=make_mock_client(return_value=mock_interaction))
    resp = run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))
    assert resp.status is None
    assert resp.finish_reason is None
    assert resp.prompt_tokens is None
    assert resp.completion_tokens is None

def test_settings_gemini_defaults_and_validation():
    """Verify default Gemini settings, valid custom assignments, and invalid rejections."""
    from backend.app.core.config import Settings

    # Defaults
    s = Settings()
    assert s.gemini_max_output_tokens == 2048
    assert s.gemini_thinking_level == "minimal"
    assert s.gemini_timeout == 45.0

    # Custom valid values
    s_custom = Settings(gemini_max_output_tokens=4096, gemini_thinking_level="low", gemini_timeout=60.0)
    assert s_custom.gemini_max_output_tokens == 4096
    assert s_custom.gemini_thinking_level == "low"
    assert s_custom.gemini_timeout == 60.0

    # Null / empty thinking level
    s_none = Settings(gemini_thinking_level="none")
    assert s_none.gemini_thinking_level is None

    # Invalid tokens
    with pytest.raises(ValueError):
        Settings(gemini_max_output_tokens=0)
    with pytest.raises(ValueError):
        Settings(gemini_max_output_tokens=-100)
    with pytest.raises(ValueError):
        Settings(gemini_max_output_tokens="invalid_number")

    # Invalid thinking level
    with pytest.raises(ValueError):
        Settings(gemini_thinking_level="ultra_high")

    # Invalid timeout
    with pytest.raises(ValueError):
        Settings(gemini_timeout=0)
    with pytest.raises(ValueError):
        Settings(gemini_timeout=-10.0)
    with pytest.raises(ValueError):
        Settings(gemini_timeout="not_a_float")


def test_gemini_provider_passes_thinking_level_and_2048_budget_to_sdk():
    """GeminiProvider passes thinking_level='minimal' and default 2048 budget to SDK generation_config."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "ok"}',
        status="completed",
        model="gemini-3.6-flash",
        steps=[],
    )
    mock_client = make_mock_client(return_value=mock_interaction)
    provider = GeminiProvider(api_key="test-api-key", model="gemini-3.6-flash", client=mock_client)

    run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))

    call_args = mock_client.aio.interactions.create.call_args
    assert call_args is not None
    kwargs = call_args.kwargs
    assert "generation_config" in kwargs
    gen_cfg = kwargs["generation_config"]
    max_out = getattr(gen_cfg, "max_output_tokens", None) or (gen_cfg.get("max_output_tokens") if isinstance(gen_cfg, dict) else None)
    thinking = getattr(gen_cfg, "thinking_level", None) or (gen_cfg.get("thinking_level") if isinstance(gen_cfg, dict) else None)
    assert max_out == 2048
    assert thinking == "minimal"


def test_gemini_provider_custom_thinking_level_and_max_tokens():
    """GeminiProvider respects per-request thinking_level and max_tokens overrides."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "ok"}',
        status="completed",
        model="gemini-3.6-flash",
        steps=[],
    )
    mock_client = make_mock_client(return_value=mock_interaction)
    provider = GeminiProvider(api_key="test-api-key", model="gemini-3.6-flash", client=mock_client)

    run_sync(provider.generate(
        [LLMMessage(role="user", content="Hi")],
        max_tokens=4096,
        thinking_level="low",
    ))

    call_args = mock_client.aio.interactions.create.call_args
    kwargs = call_args.kwargs
    gen_cfg = kwargs["generation_config"]
    max_out = getattr(gen_cfg, "max_output_tokens", None) or (gen_cfg.get("max_output_tokens") if isinstance(gen_cfg, dict) else None)
    thinking = getattr(gen_cfg, "thinking_level", None) or (gen_cfg.get("thinking_level") if isinstance(gen_cfg, dict) else None)
    assert max_out == 4096
    assert thinking == "low"


def test_gemini_provider_passes_response_format_to_sdk():
    """GeminiProvider passes response_format directly to the SDK request kwargs."""
    mock_interaction = SimpleNamespace(
        output_text='{"answer": "ok"}',
        status="completed",
        model="gemini-3.6-flash",
        steps=[],
    )
    mock_client = make_mock_client(return_value=mock_interaction)
    provider = GeminiProvider(api_key="test-api-key", model="gemini-3.6-flash", client=mock_client)

    test_rf = {"type": "text", "mime_type": "application/json", "schema": {"type": "object"}}
    run_sync(provider.generate(
        [LLMMessage(role="user", content="Hi")],
        response_format=test_rf,
    ))

    call_args = mock_client.aio.interactions.create.call_args
    kwargs = call_args.kwargs
    assert "response_format" in kwargs
    assert kwargs["response_format"] == test_rf

def test_gemini_provider_defaults_to_settings_timeout():
    """GeminiProvider defaults to settings.gemini_timeout (45.0s) and passes it to SDK interactions.create."""
    mock_interaction = make_mock_interaction(output_text="Test timeout default")
    mock_client = make_mock_client(return_value=mock_interaction)
    provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash", client=mock_client)
    assert provider.timeout == 45.0
    run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))
    assert mock_client.aio.interactions.create.call_args.kwargs["timeout"] == 45.0


def test_gemini_provider_outer_grace_timeout_descriptive_message():
    """When outer timeout fires, it raises a descriptive, sanitized TimeoutError with deadline."""
    mock_client = make_mock_client(
        side_effect=TimeoutError("Gemini API request exceeded outer deadline (50.0s)")
    )
    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-3.6-flash",
        client=mock_client,
        timeout=45.0,
        max_retries=0,
    )
    with pytest.raises(TimeoutError) as exc_info:
        run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))
    assert "exceeded outer deadline" in str(exc_info.value)
    assert "test-key" not in str(exc_info.value)


def test_gemini_provider_outer_grace_allows_sdk_transport_timeout_first():
    """SDK transport timeout is received directly before outer timeout cancels it."""
    import httpx
    req = httpx.Request("POST", "http://test")
    transport_err = httpx.ReadTimeout("Read timed out on socket", request=req)
    mock_client = make_mock_client(side_effect=transport_err)
    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-3.6-flash",
        client=mock_client,
        timeout=45.0,
        max_retries=0,
    )
    with pytest.raises(httpx.ReadTimeout):
        run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))


def test_is_retryable_error_api_timeout_error():
    """
    APITimeoutError classification regression test:
    Verify that APITimeoutError from google.genai._gaos.lib.compat_errors
    or any exception named APITimeoutError is classified as retryable.
    """
    mock_client = make_mock_client()
    provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash", client=mock_client)

    try:
        from google.genai._gaos.lib.compat_errors import APITimeoutError
        timeout_exc = APITimeoutError("Request timed out")
    except ImportError:
        APITimeoutError = type("APITimeoutError", (Exception,), {})
        timeout_exc = APITimeoutError("Request timed out")

    is_retryable, reason = provider._is_retryable_error(timeout_exc)
    assert is_retryable is True
    assert "APITimeoutError" in reason or "timed out" in reason.lower()


def test_api_timeout_error_exhausts_retries_and_verifier_fails_closed():
    """
    Verify that when generate() encounters persistent APITimeoutError,
    retries are attempted up to max_retries, and CitationVerifier fails closed
    with ClaimVerificationStatus.UNVERIFIED.
    """
    try:
        from google.genai._gaos.lib.compat_errors import APITimeoutError
    except ImportError:
        APITimeoutError = type("APITimeoutError", (Exception,), {})

    timeout_exc = APITimeoutError("Request timed out after 45.0s")
    mock_client = make_mock_client(side_effect=timeout_exc)

    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-3.6-flash",
        client=mock_client,
        timeout=45.0,
        max_retries=2,
        initial_delay=0.01,
        backoff_factor=1.0,
    )

    from backend.app.citations.verifier import CitationVerifier
    from backend.app.schemas import ClaimVerificationStatus

    chunk = SimpleNamespace(
        chunk_id="chk_timeout",
        doc_id="doc_test",
        page=1,
        text="LEMS uses integer linear programming for rank search.",
    )

    verifier = CitationVerifier(llm_provider=provider)
    check = run_sync(verifier.verify_citation(
        claim_id="c_timeout",
        claim_text="LEMS uses integer linear programming.",
        chunk_id="chk_timeout",
        quote="integer linear programming",
        retrieved_chunks=[chunk],
    ))

    # SDK interactions.create must have been called 1 + max_retries = 3 times
    assert mock_client.aio.interactions.create.call_count == 3
    # Verifier must fail closed as UNVERIFIED
    assert check.status == ClaimVerificationStatus.UNVERIFIED
    assert check.supports is None
    assert "APITimeoutError" in check.notes or "timed out" in check.notes.lower()


def test_is_retryable_error_daily_quota_exhaustion_is_non_retryable():
    """
    Daily quota exhaustion (e.g. 20 requests/day exceeded on free tier)
    must be classified as non-retryable so that futile retries are aborted immediately.
    """
    from google.genai.errors import ClientError

    quota_err = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota). Free tier limit of 20 requests/day exceeded.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    mock_client = make_mock_client(side_effect=quota_err)
    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-3.6-flash",
        client=mock_client,
        max_retries=3,
        initial_delay=0.01,
    )

    is_retryable, reason = provider._is_retryable_error(quota_err)
    assert is_retryable is False
    assert "Daily quota exhausted" in reason

    # Calling generate() should fail on attempt 1 without retrying 3 times
    with pytest.raises(ClientError):
        run_sync(provider.generate([LLMMessage(role="user", content="Hi")]))

    assert mock_client.aio.interactions.create.call_count == 1


def test_is_retryable_error_transient_429_remains_retryable():
    """
    Transient HTTP 429 errors (e.g. per-minute rate limit bursts)
    must remain retryable under the bounded retry policy.
    """
    from google.genai.errors import ClientError

    transient_err = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded. Too many requests per minute. Please retry after 2s.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    mock_client = make_mock_client()
    provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash", client=mock_client)

    is_retryable, reason = provider._is_retryable_error(transient_err)
    assert is_retryable is True
    assert "Rate limit (HTTP 429)" in reason


def test_citation_verifier_surfaces_clear_quota_diagnostic():
    """
    When semantic entailment verification fails due to daily quota exhaustion,
    CitationVerifier must surface a clear, sanitized diagnostic note indicating
    that semantic verification could not complete because of quota exhaustion,
    and fail closed as UNVERIFIED.
    """
    from google.genai.errors import ClientError
    from backend.app.citations.verifier import CitationVerifier
    from backend.app.schemas import ClaimVerificationStatus

    quota_err = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota). Free tier limit of 20 requests/day exceeded.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    mock_client = make_mock_client(side_effect=quota_err)
    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-3.6-flash",
        client=mock_client,
        max_retries=0,
    )

    verifier = CitationVerifier(llm_provider=provider)
    chunk = SimpleNamespace(
        chunk_id="chk_q",
        doc_id="doc_q",
        page=1,
        text="LEMS introduces an ILP formulation.",
    )

    check = run_sync(verifier.verify_citation(
        claim_id="c_quota",
        claim_text="LEMS uses ILP.",
        chunk_id="chk_q",
        quote="LEMS introduces an ILP formulation",
        retrieved_chunks=[chunk],
    ))

    assert check.status == ClaimVerificationStatus.UNVERIFIED
    assert check.supports is None
    assert "Gemini daily quota was exhausted" in check.notes


def test_rate_limit_error_with_quota_in_body_is_non_retryable():
    """
    When Google GenAI SDK raises RateLimitError with quota details located in exc.body,
    it must be recognized as daily quota exhaustion and classified as non-retryable.
    """
    from google.genai._gaos.lib import compat_errors as gaos_compat_errors
    import httpx

    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    err = gaos_compat_errors.RateLimitError(
        message="Resource has been exhausted",
        response=resp,
        body={
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota). Please wait 21h0m0s before retrying.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    mock_client = make_mock_client()
    provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash", client=mock_client)

    is_retryable, reason = provider._is_retryable_error(err)
    assert is_retryable is False
    assert reason == "Daily quota exhausted (HTTP 429)"


def test_daily_quota_exhaustion_fails_fast_on_attempt_1_without_retries():
    """
    Daily quota exhaustion (e.g. wait 21 hours) must fail immediately on attempt 1
    without executing futile retries or backoff delays.
    """
    from google.genai._gaos.lib import compat_errors as gaos_compat_errors
    import httpx

    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    quota_err = gaos_compat_errors.RateLimitError(
        message="Resource has been exhausted (e.g. check quota). Please wait 21 hours before retrying.",
        response=resp,
        body={
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota). Please wait 21 hours before retrying.",
            }
        },
    )

    mock_client = make_mock_client(side_effect=quota_err)
    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-3.6-flash",
        client=mock_client,
        max_retries=3,
        initial_delay=0.01,
    )

    with pytest.raises(gaos_compat_errors.RateLimitError):
        run_sync(provider.generate([
            LLMMessage(role="user", content="Research question")
        ]))

    # Strict fail-fast: must have failed on attempt 1, never retried
    assert mock_client.aio.interactions.create.await_count == 1


def test_transient_per_minute_rate_limit_remains_retryable():
    """
    Genuine transient rate limits (such as per-minute limits) must retain bounded retries.
    """
    from google.genai._gaos.lib import compat_errors as gaos_compat_errors
    import httpx

    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    transient_err = gaos_compat_errors.RateLimitError(
        message="Rate limit exceeded: 15 requests per minute. Please retry after 20s.",
        response=resp,
        body={"error": {"message": "Rate limit exceeded: 15 requests per minute. Please retry after 20s."}},
    )

    mock_client = make_mock_client()
    provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash", client=mock_client)

    is_retryable, reason = provider._is_retryable_error(transient_err)
    assert is_retryable is True
    assert reason == "Rate limit (HTTP 429)"


def test_generic_resource_exhausted_without_daily_indicators_remains_retryable():
    """
    Generic 'Resource has been exhausted' or HTTP 429 without explicit daily indicators
    or long reset windows must not be falsely treated as daily quota exhaustion.
    """
    from google.genai._gaos.lib import compat_errors as gaos_compat_errors
    import httpx

    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    generic_err = gaos_compat_errors.RateLimitError(
        message="Resource has been exhausted (e.g. check quota).",
        response=resp,
        body={"error": {"message": "Resource has been exhausted (e.g. check quota)."}},
    )

    mock_client = make_mock_client()
    provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash", client=mock_client)

    is_retryable, reason = provider._is_retryable_error(generic_err)
    assert is_retryable is True
    assert reason == "Rate limit (HTTP 429)"


def test_unrelated_error_with_hours_never_misclassified_as_daily_quota():
    """
    Arbitrary errors mentioning 'hours' without quota context must never trigger daily quota classification.
    """
    from backend.app.llm.gemini import is_daily_quota_exhausted

    err = RuntimeError("The batch job has been running for 3 hours.")
    assert is_daily_quota_exhausted(err) is False
