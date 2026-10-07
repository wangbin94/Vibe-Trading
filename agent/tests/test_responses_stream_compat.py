"""Regression tests for OpenAI-compatible Responses event streams."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
from langchain_core.messages import HumanMessage

from src.providers.llm import ChatOpenAIWithReasoning


class _MappingResponses:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    @contextmanager
    def create(self, **kwargs: Any) -> Iterator[Iterator[dict[str, Any]]]:
        yield iter(self._events)


class _MappingRootClient:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.responses = _MappingResponses(events)


class _MappingAsyncStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def __aenter__(self) -> "_MappingAsyncStream":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def __aiter__(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        for event in self._events:
            yield event


class _MappingAsyncResponses:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def create(self, **kwargs: Any) -> _MappingAsyncStream:
        return _MappingAsyncStream(self._events)


class _MappingAsyncRootClient:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.responses = _MappingAsyncResponses(events)


def _text_delta_event() -> dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "delta": "hello",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
    }


def _completed_event() -> dict[str, Any]:
    return {"type": "response.completed", "response": {
        "id": "resp_1", "object": "response", "created_at": 1, "status": "completed",
        "model": "gateway-reasoning-model", "output": [], "tools": [], "tool_choice": "auto",
        "parallel_tool_calls": True, "error": None, "incomplete_details": None,
        "instructions": None, "metadata": {}, "temperature": 1, "top_p": 1,
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }}


@pytest.mark.skipif(
    ChatOpenAIWithReasoning is None,
    reason="langchain-openai is not installed",
)
def test_responses_stream_accepts_mapping_events() -> None:
    """OpenAI-compatible gateways may yield dicts instead of SDK event objects."""
    events = [_text_delta_event(), _completed_event()]
    llm = ChatOpenAIWithReasoning(
        model="gateway-reasoning-model",
        api_key="sk-test",
        use_responses_api=True,
    )
    llm.root_client = _MappingRootClient(events)
    # Keep this test focused on stream-event compatibility rather than request
    # serialization, which is exercised by the provider payload tests.
    llm._get_request_payload = lambda *args, **kwargs: {
        "model": "gateway-reasoning-model",
        "input": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    chunks = list(llm.stream("hello"))

    assert "".join(chunk.text for chunk in chunks) == "hello"


@pytest.mark.skipif(
    ChatOpenAIWithReasoning is None,
    reason="langchain-openai is not installed",
)
def test_async_responses_stream_accepts_mapping_events() -> None:
    """The async Responses path must normalize gateway mappings as well."""
    llm = ChatOpenAIWithReasoning(
        model="gateway-reasoning-model",
        api_key="sk-test",
        use_responses_api=True,
    )
    llm.root_async_client = _MappingAsyncRootClient([_text_delta_event(), _completed_event()])
    llm._get_request_payload = lambda *args, **kwargs: {
        "model": "gateway-reasoning-model",
        "input": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    async def collect() -> str:
        chunks = [chunk async for chunk in llm.astream("hello")]
        return "".join(chunk.text for chunk in chunks)

    assert asyncio.run(collect()) == "hello"


@pytest.mark.skipif(
    ChatOpenAIWithReasoning is None,
    reason="langchain-openai is not installed",
)
def test_responses_payload_does_not_assume_chat_messages() -> None:
    """Responses requests use ``input`` and must bypass chat-only rewriting."""
    llm = ChatOpenAIWithReasoning(
        model="gateway-reasoning-model",
        api_key="sk-test",
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "high"},
    )

    payload = llm._get_request_payload([HumanMessage(content="hello")])

    assert payload["input"]
    assert payload["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("event_type", ["response.failed", "error"])
@pytest.mark.parametrize("typed", [False, True])
def test_failed_response_events_are_not_silently_dropped(event_type: str, typed: bool) -> None:
    from types import SimpleNamespace
    from src.providers.llm import ResponsesStreamFailure, _normalize_responses_stream_event

    error = {"code": "server_error", "message": "upstream unavailable"}
    event = {"type": event_type, "response": {"id": "resp_failed", "error": error}}
    if typed:
        event = SimpleNamespace(type=event_type, response=SimpleNamespace(id="resp_failed", error=SimpleNamespace(**error)))
    with pytest.raises(ResponsesStreamFailure, match="server_error: upstream unavailable") as caught:
        _normalize_responses_stream_event(event)
    assert caught.value.status_code == 500
    assert caught.value.response_id == "resp_failed"


def test_failed_response_stream_preserves_provider_error_classification() -> None:
    from src.providers.chat import ChatLLM, ProviderStreamError

    llm = ChatOpenAIWithReasoning(model="gateway-model", api_key="sk-test", use_responses_api=True)
    llm.root_client = _MappingRootClient([
        {"type": "response.failed", "response": {"error": {"code": "invalid_request_error", "message": "bad input"}}},
    ])
    client = ChatLLM.__new__(ChatLLM)
    client.model_name = "gateway-model"
    client._llm = llm
    with pytest.raises(ProviderStreamError, match="responses_stream_failed") as caught:
        client.stream_chat([{"role": "user", "content": "hello"}])
    assert caught.value.status_code == 400
    assert caught.value.retryable is False


def test_async_failed_response_stream_raises_original_failure() -> None:
    from src.providers.llm import ResponsesStreamFailure

    llm = ChatOpenAIWithReasoning(model="gateway-model", api_key="sk-test", use_responses_api=True)
    llm.root_async_client = _MappingAsyncRootClient([
        {"type": "response.failed", "response": {"error": {"code": "server_error", "message": "upstream unavailable"}}},
    ])

    async def collect():
        return [chunk async for chunk in llm.astream("hello")]

    with pytest.raises(ResponsesStreamFailure, match="upstream unavailable"):
        asyncio.run(collect())


@pytest.mark.parametrize("events", [[], [{"type": "response.created", "response": {"id": "resp_1"}}], [_text_delta_event()]])
def test_sync_truncated_response_stream_is_not_accepted(events) -> None:
    from src.providers.llm import _ResponsesSyncStream, ResponsesStreamFailure
    with _MappingResponses(events).create() as stream:
        with pytest.raises(ResponsesStreamFailure, match="stream_incomplete"):
            list(_ResponsesSyncStream(stream))


def test_async_truncated_response_stream_is_not_accepted() -> None:
    from src.providers.llm import _ResponsesAsyncStream, ResponsesStreamFailure
    async def collect():
        stream = _ResponsesAsyncStream(_MappingAsyncStream([_text_delta_event()]))
        return [event async for event in stream]
    with pytest.raises(ResponsesStreamFailure, match="stream_incomplete"):
        asyncio.run(collect())


def _function_events(count=2, *, deltas=False, terminal_only=False):
    import json
    events=[]
    items=[]
    for index in range(count):
        arguments=json.dumps({'source_id':f'page_{index}'})
        item={'type':'function_call','id':f'fc_{index}','call_id':f'call_{index}','name':'frozen_source','arguments':arguments,'status':'completed'}
        items.append(item)
        events.append({'type':'response.output_item.added','output_index':index,'item':{**item,'arguments':''}})
        if deltas:
            events.append({'type':'response.function_call_arguments.delta','output_index':index,'delta':arguments[:10]})
            events.append({'type':'response.function_call_arguments.delta','output_index':index,'delta':arguments[10:]})
        if not terminal_only:
            events.append({'type':'response.function_call_arguments.done','output_index':index,'arguments':arguments})
            events.append({'type':'response.output_item.done','output_index':index,'item':item})
    end=_completed_event()
    end['response']['output']=items
    return [*events,end]


@pytest.mark.parametrize('deltas',[False,True])
@pytest.mark.parametrize('terminal_only',[False,True])
def test_complete_function_arguments_survive_done_and_terminal_events(deltas,terminal_only):
    llm=ChatOpenAIWithReasoning(model='gateway-model',api_key='sk-test',use_responses_api=True)
    llm.root_client=_MappingRootClient(_function_events(62,deltas=deltas,terminal_only=terminal_only))
    chunks=list(llm.stream('read frozen sources'))
    merged=chunks[0]
    for chunk in chunks[1:]: merged+=chunk
    assert len(merged.tool_calls)==62
    assert [call['args'] for call in merged.tool_calls]==[{'source_id':f'page_{i}'} for i in range(62)]
    assert [call['id'] for call in merged.tool_calls]==[f'call_{i}' for i in range(62)]


def test_async_complete_function_arguments_survive_typed_done_only_events():
    from src.providers.llm import _ResponsesMappingEvent
    events=[_ResponsesMappingEvent({**e,**({'item':_ResponsesMappingEvent(e['item'])} if 'item' in e else {})}) for e in _function_events()]
    llm=ChatOpenAIWithReasoning(model='gateway-model',api_key='sk-test',use_responses_api=True)
    llm.root_async_client=_MappingAsyncRootClient(events)
    async def collect():
        chunks=[chunk async for chunk in llm.astream('read frozen sources')]
        merged=chunks[0]
        for chunk in chunks[1:]:merged+=chunk
        return merged
    merged=asyncio.run(collect())
    assert [call['args'] for call in merged.tool_calls]==[{'source_id':'page_0'},{'source_id':'page_1'}]


@pytest.mark.parametrize('conflict',[False,True])
def test_missing_or_conflicting_function_arguments_fail_before_empty_calls(conflict):
    from src.providers.llm import _ResponsesSyncStream,ResponsesStreamFailure
    events=_function_events(1)
    if conflict:
        events.insert(1,{'type':'response.function_call_arguments.delta','output_index':0,'delta':'{"other":'})
    else:
        events[1]['arguments']=''
        events[2]['item']['arguments']=''
        events[3]['response']['output'][0]['arguments']=''
    with _MappingResponses(events).create() as stream:
        with pytest.raises(ResponsesStreamFailure,match='tool_arguments_'):
            list(_ResponsesSyncStream(stream))
