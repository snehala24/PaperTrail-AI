from abc import ABC, abstractmethod
from typing import Any, List, Optional
from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    thought_tokens: Optional[int] = None


class LLMProvider(ABC):
    """
    Abstract LLM Provider interface.
    Provider-specific code (e.g. Google Gemini) must reside exclusively
    in backend/app/llm and implement this interface.
    """

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the language model given a list of messages."""
        pass
