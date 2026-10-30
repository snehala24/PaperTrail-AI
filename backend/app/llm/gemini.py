"""
Google Gemini LLM Provider implementation for PaperTrail AI.

Uses the official google-genai SDK's Interactions API
(client.aio.interactions.create) for asynchronous text generation.
"""

import asyncio
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.config import settings
from backend.app.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


def extract_error_text_payload(err: Exception) -> str:
    """Extract and concatenate text representations from all available SDK exception attributes."""
    parts = [str(err)]
    msg = getattr(err, "message", None)
    if isinstance(msg, str) and msg:
        parts.append(msg)
    body = getattr(err, "body", None)
    if body is not None:
        parts.append(str(body))
    resp_json = getattr(err, "response_json", None)
    if resp_json is not None:
        parts.append(str(resp_json))
    args = getattr(err, "args", None)
    if args:
        for a in args:
            if a:
                parts.append(str(a))
    resp = getattr(err, "response", None)
    if resp is not None:
        resp_text = getattr(resp, "text", None)
        if isinstance(resp_text, str) and resp_text:
            parts.append(resp_text)
    return " ".join(parts)


def is_daily_quota_exhausted(err: Exception) -> bool:
    """
    Detect whether an exception indicates daily quota exhaustion or an extended quota reset window.
    Reliably distinguishes daily quota limits from transient per-minute rate limits.
    """
    text = extract_error_text_payload(err).lower()

    # Supporting quota context: must be present so arbitrary text (e.g. 'office hours') never matches
    has_quota_context = any(
        ctx in text
        for ctx in ("quota", "limit", "exhaust", "rate", "429", "resource")
    )
    if not has_quota_context:
        return False

    # 1. Direct explicit daily quota / free-tier indicators
    daily_direct_keywords = (
        "requests/day",
        "request/day",
        "per day",
        "per_day",
        "daily",
        "limit of 20 requests",
        "free-tier limit",
        "free_tier limit",
        "free tier limit",
        "generate_content_requests_per_day",
        "requests_per_day",
    )
    if any(k in text for k in daily_direct_keywords):
        return True

    # 2. Extended reset window indicating hours (e.g., 'wait 21h0m0s', 'wait 21 hours', 'retry after 18h')
    has_long_wait_window = bool(
        re.search(
            r"(?:wait|retry[ _-]?after|reset in)\s*(?:approx(?:imately)?\s*)?(\d+)\s*(?:h(?:ours?|rs?|\b|\d)|hr\b)",
            text,
        )
    )
    if has_long_wait_window:
        return True

    return False


def _supports_temperature(model_name: str) -> bool:
    """
    Check whether a model supports custom temperature settings.
    Gemini 4 and Gemini 3.6+ models ignore custom values for temperature,
    top_p, and top_k and return deprecation warnings.
    """
    lower = model_name.lower()
    for prefix in ("gemini-4", "gemini-3.6", "gemini-3.7", "gemini-3.8"):
        if prefix in lower:
            return False
    return True


class GeminiProvider(LLMProvider):
    """
    Hosted LLM Provider for Google Gemini using the official google-genai SDK.

    Implements the abstract LLMProvider interface with per-request timeouts,
    bounded retries with exponential backoff and jitter for transient errors,
    and safe observability logging.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        thinking_level: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        resolved_timeout = (
            timeout
            if timeout is not None
            else getattr(settings, "gemini_timeout", 45.0)
        )
        if resolved_timeout <= 0:
            raise ValueError("timeout must be positive")
        resolved_max_retries = (
            int(os.environ["GEMINI_MAX_RETRIES"])
            if "GEMINI_MAX_RETRIES" in os.environ
            else max_retries
        )
        if resolved_max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if initial_delay <= 0 or max_delay <= 0:
            raise ValueError("retry delays must be positive")
        if backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")

        self.timeout = float(resolved_timeout)
        self.max_retries = resolved_max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        # Load API key: explicitly passed or from settings
        resolved_key = api_key if api_key is not None else settings.gemini_api_key
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not set in environment or configuration. Please configure GEMINI_API_KEY."
            )
        self._api_key = resolved_key.strip()

        # Load Model: explicitly passed or from settings
        resolved_model = model if model is not None else settings.gemini_model
        if not resolved_model or not resolved_model.strip():
            raise ValueError(
                "GEMINI_MODEL is not set in environment or configuration. Please configure GEMINI_MODEL."
            )
        self.model = resolved_model.strip()

        # Configure thinking_level: if explicitly passed, use it;
        # otherwise for gemini-3.6 models, default to configured gemini_thinking_level
        if thinking_level is not None:
            self.thinking_level = thinking_level
        elif "gemini-3.6" in self.model.lower():
            self.thinking_level = settings.gemini_thinking_level
        else:
            self.thinking_level = None

        if client is not None:
            self._client = client
        else:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)

        # Disable SDK-internal retries to prevent multiplying provider retries
        # with SDK retries (which would otherwise allow up to 4x4 = 16 attempts).
        try:
            from google.genai._gaos.utils.retries import RetryConfig

            aio = getattr(self._client, "aio", None)
            interactions = getattr(aio, "interactions", None)
            cfg = getattr(interactions, "sdk_configuration", None)
            if cfg and hasattr(cfg, "retry_config"):
                cfg.retry_config = RetryConfig(
                    strategy="none",
                    backoff=None,
                    retry_connection_errors=False,
                    max_retries=0,
                )
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"GeminiProvider(model={self.model!r}, timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )

    def _map_messages(
        self,
        messages: List[LLMMessage],
    ) -> Tuple[Optional[str], Any]:
        """
        Validate messages and map them to system_instruction and input format.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        supported_roles = {"system", "user", "assistant"}
        system_parts: List[str] = []
        conversation: List[LLMMessage] = []

        for idx, msg in enumerate(messages):
            if not isinstance(msg, LLMMessage):
                raise TypeError(
                    f"Expected LLMMessage at index {idx}, got {type(msg).__name__}"
                )
            if msg.role not in supported_roles:
                raise ValueError(
                    f"Unsupported message role: {msg.role!r}. Supported roles: {sorted(supported_roles)}"
                )
            if msg.role == "system":
                if msg.content.strip():
                    system_parts.append(msg.content.strip())
            else:
                conversation.append(msg)

        if not conversation:
            raise ValueError(
                "Messages list must contain at least one user or assistant message"
            )

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        # For a single user turn, pass as string directly
        if len(conversation) == 1 and conversation[0].role == "user":
            input_payload: Any = conversation[0].content
        else:
            # Multi-turn interaction steps
            steps: List[Dict[str, Any]] = []
            for m in conversation:
                if m.role == "user":
                    steps.append({
                        "type": "user_input",
                        "content": [{"type": "text", "text": m.content}],
                    })
                elif m.role == "assistant":
                    steps.append({
                        "type": "model_output",
                        "content": [{"type": "text", "text": m.content}],
                    })
            input_payload = steps

        return system_instruction, input_payload

    def _is_retryable_error(self, exc: Exception) -> Tuple[bool, str]:
        """Determine whether an error is transient and retryable."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            msg = str(exc).strip()
            return True, msg if msg else "Request timed out"

        try:
            from google.genai import errors as genai_errors
        except ImportError:
            genai_errors = None

        try:
            from google.genai._gaos.lib import compat_errors as gaos_compat_errors
        except ImportError:
            gaos_compat_errors = None

        try:
            import httpx
        except ImportError:
            httpx = None

        # Check for APITimeoutError from google-genai SDK
        if (
            (gaos_compat_errors and hasattr(gaos_compat_errors, "APITimeoutError") and isinstance(exc, gaos_compat_errors.APITimeoutError))
            or (genai_errors and hasattr(genai_errors, "APITimeoutError") and isinstance(exc, genai_errors.APITimeoutError))
            or exc.__class__.__name__ == "APITimeoutError"
        ):
            msg = str(exc).strip()
            return True, msg if msg else "Request timed out (APITimeoutError)"

        # Check for explicit daily quota exhaustion across all exception representations
        if is_daily_quota_exhausted(exc):
            return False, "Daily quota exhausted (HTTP 429)"

        # Check for RateLimitError from google-genai GAOS SDK
        if (
            (gaos_compat_errors and hasattr(gaos_compat_errors, "RateLimitError") and isinstance(exc, gaos_compat_errors.RateLimitError))
            or exc.__class__.__name__ == "RateLimitError"
        ):
            return True, "Rate limit (HTTP 429)"

        # Check whether exc is an SDK or HTTP API error
        is_sdk_error = False
        if genai_errors and isinstance(exc, genai_errors.APIError):
            is_sdk_error = True
        elif gaos_compat_errors and isinstance(exc, gaos_compat_errors.APIError):
            is_sdk_error = True
        elif exc.__class__.__module__.startswith("google.genai.") and (
            hasattr(exc, "status_code") or hasattr(exc, "code")
        ):
            is_sdk_error = True

        is_httpx_status = bool(httpx and isinstance(exc, httpx.HTTPStatusError))

        if is_sdk_error or is_httpx_status:
            # Check explicit ServerError / ClientError subclasses from genai_errors
            if genai_errors and isinstance(exc, genai_errors.ServerError):
                code = getattr(exc, "code", None)
                return True, f"Server error (HTTP {code})"

            if genai_errors and isinstance(exc, genai_errors.ClientError):
                code = getattr(exc, "code", None)
                if code == 429:
                    return True, "Rate limit (HTTP 429)"
                return False, f"Permanent client error (HTTP {code})"

            # Extract numeric status code
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(exc, "code", None)
            if status_code is None and is_httpx_status and hasattr(exc, "response"):
                status_code = getattr(exc.response, "status_code", None)

            if isinstance(status_code, int):
                if status_code == 429:
                    return True, "Rate limit (HTTP 429)"
                if status_code in {500, 502, 503, 504}:
                    return True, f"Server error (HTTP {status_code})"
                if 400 <= status_code < 500:
                    return False, f"Permanent client error (HTTP {status_code})"
                if 500 <= status_code < 600:
                    return True, f"Server error (HTTP {status_code})"
                return False, f"API error (HTTP {status_code})"

        if httpx and isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return True, f"Transport/network error ({type(exc).__name__})"

        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True, f"Connection error ({type(exc).__name__})"

        return False, f"Non-retryable exception ({type(exc).__name__})"

    def _build_response(
        self,
        interaction: Any,
        attempt: int,
        start_time: float,
    ) -> LLMResponse:
        """Parse interaction output, extract tokens, log safe metadata, and return LLMResponse."""
        if hasattr(interaction, "errors") and interaction.errors:
            err_msgs = "; ".join(
                f"{getattr(e, 'code', '')}: {getattr(e, 'message', '')}".strip(": ")
                for e in interaction.errors
            )
            raise RuntimeError(f"Gemini API returned failed interaction: {err_msgs}")

        content = getattr(interaction, "output_text", None)
        if not content and hasattr(interaction, "steps") and interaction.steps:
            parts: List[str] = []
            for step in interaction.steps:
                step_type = getattr(step, "type", None)
                if step_type == "model_output":
                    step_content = getattr(step, "content", None)
                    if step_content:
                        for c in step_content:
                            text = getattr(c, "text", None)
                            if text:
                                parts.append(text)
            if parts:
                content = "".join(parts)

        if content is None or not content.strip():
            raise RuntimeError("Received empty response from Gemini API")

        status = getattr(interaction, "status", None)
        usage = getattr(interaction, "usage", None)
        prompt_tokens = getattr(usage, "total_input_tokens", None) if usage else None
        completion_tokens = (
            getattr(usage, "total_output_tokens", None) if usage else None
        )
        thought_tokens = (
            getattr(usage, "total_thought_tokens", None) if usage else None
        )
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        # Determine finish reason from interaction status and token limits
        finish_reason: Optional[str] = None
        if status == "budget_exceeded":
            finish_reason = "max_tokens"
        elif (
            completion_tokens is not None
            and getattr(self, "_last_max_tokens", None) is not None
            and completion_tokens >= self._last_max_tokens
        ):
            finish_reason = "max_tokens"
        elif status == "incomplete":
            finish_reason = "incomplete"
        elif status == "completed":
            finish_reason = "stop"

        model_name = getattr(interaction, "model", None) or self.model
        elapsed = time.perf_counter() - start_time

        step_types = [
            getattr(s, "type", "unknown")
            for s in (getattr(interaction, "steps", None) or [])
        ]

        logger.info(
            "Gemini generation completed (model: %s, status: %s, finish_reason: %s, "
            "step_types: %s, attempt: %d, elapsed: %.2fs, prompt_tokens: %s, "
            "completion_tokens: %s, thought_tokens: %s, total_tokens: %s)",
            model_name,
            status,
            finish_reason,
            step_types,
            attempt + 1,
            elapsed,
            prompt_tokens,
            completion_tokens,
            thought_tokens,
            total_tokens,
        )

        return LLMResponse(
            content=content.strip(),
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status=status,
            finish_reason=finish_reason,
            thought_tokens=thought_tokens,
        )

    async def _execute_with_timeout(self, coro: Any) -> Any:
        """Execute coroutine with per-request timeout if an event loop is running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Outer grace buffer (5.0s) ensures the SDK/transport has priority to raise
            # its own structured transport timeout before the coroutine is cancelled.
            outer_grace = 5.0 if self.timeout >= 1.0 else max(0.01, self.timeout * 0.2)
            outer_deadline = self.timeout + outer_grace
            try:
                return await asyncio.wait_for(coro, timeout=outer_deadline)
            except (asyncio.TimeoutError, TimeoutError):
                raise TimeoutError(
                    f"Gemini API request exceeded outer deadline ({outer_deadline:.1f}s)"
                )
        return await coro

    async def _sleep(self, delay: float) -> None:
        """Asynchronously sleep if an event loop is running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            await asyncio.sleep(delay)

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Any] = None,
        thinking_level: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from Google Gemini given a list of messages.
        """
        if temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        effective_max_tokens = (
            max_tokens if max_tokens is not None else settings.gemini_max_output_tokens
        )
        self._last_max_tokens = effective_max_tokens
        system_instruction, input_payload = self._map_messages(messages)

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_payload,
        }
        if system_instruction is not None:
            call_kwargs["system_instruction"] = system_instruction
        if response_format is not None:
            call_kwargs["response_format"] = response_format

        gen_config: Dict[str, Any] = {}
        if effective_max_tokens is not None:
            gen_config["max_output_tokens"] = effective_max_tokens

        effective_thinking = (
            thinking_level if thinking_level is not None else self.thinking_level
        )
        if effective_thinking is not None:
            gen_config["thinking_level"] = effective_thinking

        if _supports_temperature(self.model) and temperature > 0.0:
            gen_config["temperature"] = temperature
        elif not _supports_temperature(self.model) and temperature > 0.0:
            logger.info(
                "Model %s does not support custom temperature; ignoring temperature=%.2f",
                self.model,
                temperature,
            )

        if gen_config:
            call_kwargs["generation_config"] = gen_config

        start_time = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                interaction = await self._execute_with_timeout(
                    self._client.aio.interactions.create(
                        timeout=self.timeout,
                        **call_kwargs,
                    )
                )
                return self._build_response(interaction, attempt, start_time)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                is_retryable, reason = self._is_retryable_error(exc)
                elapsed = time.perf_counter() - start_time
                logger.warning(
                    "Gemini generation attempt %d/%d failed (elapsed: %.2fs, retryable: %s): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    elapsed,
                    is_retryable,
                    reason,
                )
                if not is_retryable or attempt >= self.max_retries:
                    break

                delay = min(
                    self.initial_delay * (self.backoff_factor**attempt)
                    + random.uniform(0.0, 0.25),
                    self.max_delay,
                )
                await self._sleep(delay)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "Unexpected state: generation loop exited without result or error"
        )

    async def aclose(self) -> None:
        """Explicitly close the underlying asynchronous client to release connections."""
        if self._client is not None:
            aio = getattr(self._client, "aio", None)
            if aio is not None and hasattr(aio, "aclose"):
                await aio.aclose()

    async def __aenter__(self) -> "GeminiProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
