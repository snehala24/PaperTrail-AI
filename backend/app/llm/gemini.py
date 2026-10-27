"""
Google Gemini LLM Provider implementation for PaperTrail AI.

Uses the official google-genai SDK's Interactions API
(client.aio.interactions.create) for asynchronous text generation.
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.config import settings
from backend.app.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


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
        timeout: float = 30.0,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        client: Optional[Any] = None,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if initial_delay <= 0 or max_delay <= 0:
            raise ValueError("retry delays must be positive")
        if backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")

        self.timeout = timeout
        self.max_retries = max_retries
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
        if isinstance(exc, asyncio.TimeoutError):
            return True, "Request timed out"

        try:
            from google.genai import errors as genai_errors

            if isinstance(exc, genai_errors.ClientError):
                code = getattr(exc, "code", None)
                if code == 429:
                    return True, "Rate limit (HTTP 429)"
                return False, f"Permanent client error (HTTP {code})"

            if isinstance(exc, genai_errors.ServerError):
                code = getattr(exc, "code", None)
                return True, f"Server error (HTTP {code})"

            if isinstance(exc, genai_errors.APIError):
                code = getattr(exc, "code", None)
                if code in {408, 429, 500, 502, 503, 504}:
                    return True, f"Transient API error (HTTP {code})"
                return False, f"Permanent API error (HTTP {code})"
        except ImportError:
            pass

        try:
            import httpx

            if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
                return True, f"Transport/network error ({type(exc).__name__})"
        except ImportError:
            pass

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

        usage = getattr(interaction, "usage", None)
        prompt_tokens = getattr(usage, "total_input_tokens", None) if usage else None
        completion_tokens = (
            getattr(usage, "total_output_tokens", None) if usage else None
        )
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        model_name = getattr(interaction, "model", None) or self.model
        elapsed = time.perf_counter() - start_time

        logger.info(
            "Gemini generation completed (model: %s, attempt: %d, elapsed: %.2fs, "
            "prompt_tokens: %s, completion_tokens: %s, total_tokens: %s)",
            model_name,
            attempt + 1,
            elapsed,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        return LLMResponse(
            content=content.strip(),
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def _execute_with_timeout(self, coro: Any) -> Any:
        """Execute coroutine with per-request timeout if an event loop is running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            return await asyncio.wait_for(coro, timeout=self.timeout)
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
    ) -> LLMResponse:
        """
        Generate a response from Google Gemini given a list of messages.
        """
        if temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        system_instruction, input_payload = self._map_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_payload,
        }
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction

        gen_config: Dict[str, Any] = {}
        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        if _supports_temperature(self.model) and temperature > 0.0:
            gen_config["temperature"] = temperature
        elif not _supports_temperature(self.model) and temperature > 0.0:
            logger.info(
                "Model %s does not support custom temperature; ignoring temperature=%.2f",
                self.model,
                temperature,
            )

        if gen_config:
            kwargs["generation_config"] = gen_config

        start_time = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                interaction = await self._execute_with_timeout(
                    self._client.aio.interactions.create(**kwargs)
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
