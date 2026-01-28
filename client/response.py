from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

@dataclass
class TextDelta:
    content: str

    def __str__(self):
        return self.content

class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens : int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

@dataclass
class StreamEvent:
    type: StreamEventType 
    text_delta: TextDelta | None = None 
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    @classmethod
    def create_error(
        cls,
        error: str,
    ) -> StreamEvent:
        return cls(
            type=StreamEventType.ERROR,
            error=error,
        )

    @classmethod
    def create_delta(
        cls,
        content: str,
    ) -> StreamEvent:
        return cls(
            type=StreamEventType.TEXT_DELTA,
            text_delta=TextDelta(content)
        )

    @classmethod
    def create_msg_complete(
        cls,
        finish_reason: str | None = None,
        usage: TokenUsage | None = None,
        content: str | None = None
    ) -> StreamEvent:
        delta = TextDelta(content) if content else None
        return cls(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
            text_delta=delta
        )
