from client.response import parse_tool_call_arguments
from client.response import ToolCall
from client.response import ToolCallDelta
from client.response import StreamEventType
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
                api_key=config.GEMINI_API_KEY,
                base_url=config.GEMINI_BASE_URL,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _build_tools(self, tools: list[dict[str, Any]]):
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool['name'],
                    'description': tool.get('description', ""),
                    'parameters': tool.get(
                        'parameters',
                        {
                            'type': 'object',
                            'properties': {}
                        }
                    )
                }
            }
            for tool in tools
        ]

    async def chat_completion(
        self, 
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool=True
    ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()

        kwargs = {
            "model": config.DEFAULT_GEMINI_MODEL,
            "messages": messages,
            "stream": stream,
            "stream_options": {"include_usage": True}
        }

        if tools:
            kwargs['tools'] = self._build_tools(tools)
            kwargs['tool_choice'] = 'auto'
    
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
        tool_calls: dict[int, dict[str, Any]] = {}

        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens if chunk.usage.prompt_tokens_details else 0,
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

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            'id': tool_call_delta.id or "",
                            'name': '',
                            'arguments': ''
                        }

                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                tool_calls[idx]['name'] = tool_call_delta.function.name
                                yield StreamEvent.create_tool_call_start(
                                    call_id=tool_calls[idx]['id'],
                                    name=tool_call_delta.function.name,
                                )

                            if tool_call_delta.function.arguments:
                                tool_calls[idx]['arguments'] += tool_call_delta.function.arguments

                                yield StreamEvent.create_tool_call_delta(
                                    call_id=tool_calls[idx]['id'],
                                    arguments=tool_call_delta.function.arguments,
                                )
                
        for index, tool_call in tool_calls.items():
            yield StreamEvent.create_tool_call_complete(
                tool_call=ToolCall(
                    call_id=tool_call['id'],
                    name=tool_call['name'],
                    arguments=parse_tool_call_arguments(tool_call['arguments'])
                )
            )

            
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

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(ToolCall(
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=parse_tool_call_arguments(tool_call.function.arguments)
                ))

        
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens if response.usage.prompt_tokens_details else 0,
            )

        return StreamEvent.create_msg_complete(finish_reason, usage, content)
      
        