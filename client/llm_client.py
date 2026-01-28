from openai import APIConnectionError, RateLimitError, AsyncOpenAI, APIError
import asyncio
from typing import Any, AsyncGenerator
from config import config
from client.response import StreamEvent, TokenUsage

class LLMClient:
    def __init__(self) -> None:
        self._client : AsyncOpenAI | None = None
        self._max_retries: int = config.MAX_RETRIES

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.BASE_URL,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self, 
        messages: list[dict[str, Any]],
        stream: bool=True
    ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()

        kwargs = {
            "model": config.DEFAULT_AI_MODEL,
            "messages": messages,
            "stream": stream
        }
    
        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return 

            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent.create_error(f"Rate Limit Error: {e}")
                    return

            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent.create_error(f"Connection error: {e}")
                    return

            except APIError as e:
                yield StreamEvent.create_error(f"API error: {e}")
                return
                
    async def _stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)

        usage: TokenUsage | None = None
        finish_reason : str | None = None

        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )

            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            content = delta.content

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if content:
                yield StreamEvent.create_delta(content)

        yield StreamEvent.create_msg_complete(finish_reason, usage)
        

    async def _non_stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: dict[str, Any]
    ) -> StreamEvent: 
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason
        
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )

        return StreamEvent.create_msg_complete(finish_reason, usage, content)
      
        