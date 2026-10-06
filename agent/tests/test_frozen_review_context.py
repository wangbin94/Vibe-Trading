"""Frozen daily evidence survives synthesis; ordinary sessions retain defaults."""
import asyncio
import json
from copy import deepcopy
from pathlib import Path
import pytest
from src.agent.loop import _bound_frozen_review_context, _microcompact, _context_collapse
from src.session.models import Attempt
from src.session.service import SessionService
from src.session.store import SessionStore
from src.session.events import EventBus


def test_complete_34_page_context_is_preserved_and_bounded():
    messages=[{'role':'system','content':'system'},{'role':'user','content':'x'*3500}]
    messages.extend({'role':'tool','name':'mcp_trading_copilot_tc_get_daily_review_evidence','content':json.dumps({'source_id':f'page_{i}','evidence':'x'*6500}),'tool_call_id':str(i)} for i in range(34))
    original=deepcopy(messages)
    _bound_frozen_review_context(messages)
    assert messages==original
    normal=deepcopy(messages);_microcompact(normal);_context_collapse(normal)
    assert normal != original
    with pytest.raises(ValueError,match='budget_exceeded'):
        _bound_frozen_review_context([{'content':'x'*320001}])


def test_daily_session_disables_unscoped_memory_and_enables_context_retention(monkeypatch,tmp_path:Path):
    class Index:
        def index_session(self,*args):pass
        def index_message(self,*args):pass
    observed={}
    class Loop:
        def __init__(self,**kwargs):observed.update(kwargs)
        def run(self,**kwargs):return {'status':'completed'}
    monkeypatch.setattr('src.session.service.get_shared_index',lambda:Index())
    from types import SimpleNamespace
    monkeypatch.setattr('src.tools.build_registry',lambda **kwargs:SimpleNamespace(get=lambda name:SimpleNamespace(name=name)))
    monkeypatch.setattr('src.providers.chat.ChatLLM',lambda:object())
    def forbidden():raise AssertionError('blind review read global personal memory')
    monkeypatch.setattr('src.memory.persistent.PersistentMemory',forbidden)
    monkeypatch.setattr('src.agent.loop.AgentLoop',Loop)
    monkeypatch.setattr('src.config.loader.load_runtime_agent_config',lambda overrides=None:object())
    service=SessionService(store=SessionStore(tmp_path/'sessions'),event_bus=EventBus(),runs_dir=tmp_path/'runs')
    result=asyncio.run(service._run_with_agent(Attempt(session_id='blind',prompt='review'),messages=[],session_config={'frozen_review_context':True}))
    assert result['status']=='completed'
    assert observed['persistent_memory'] is None and observed['frozen_review_context'] is True
    assert set(observed['registry']._tools)=={'mcp_trading_copilot_tc_get_daily_review_evidence'}


def test_agent_sees_every_prior_page_at_synthesis(monkeypatch,tmp_path:Path):
    from types import SimpleNamespace
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry, BaseTool
    class Source(BaseTool):
        name='frozen_source'
        repeatable=True
        parameters={'type':'object','properties':{'page':{'type':'integer'}},'required':['page']}
        def execute(self,**kwargs):return json.dumps({'status':'ok','page_'+str(kwargs['page']):'x'*6500})
    class Spy:
        calls=0
        def stream_chat(self,messages,**kwargs):
            pages=[m for m in messages if m['role']=='tool']
            assert all('page_'+str(i) in page['content'] for i,page in enumerate(pages))
            self.calls+=1
            if self.calls==35:
                assert len(pages)==34
                return SimpleNamespace(has_tool_calls=False,content='complete evidence synthesis')
            return SimpleNamespace(has_tool_calls=True,content=None,reasoning_content=None,tool_calls=[SimpleNamespace(id=str(self.calls),name='frozen_source',arguments={'page':self.calls-1})])
    spy=Spy();registry=ToolRegistry();registry.register(Source());agent=AgentLoop(registry=registry,llm=spy,frozen_review_context=True,max_iterations=40)
    run_dir=tmp_path/'run';run_dir.mkdir();agent.memory.run_dir=str(run_dir)
    result=agent.run('Read the entire day then synthesize')
    assert result['status']=='success', result
    assert spy.calls==35
