"""Frozen daily evidence survives synthesis; ordinary sessions retain defaults."""
import asyncio
import json
from copy import deepcopy
from pathlib import Path
import pytest
from src.agent.loop import _bound_frozen_review_context, _microcompact, _context_collapse, FROZEN_REVIEW_CONTEXT_MAX_CHARS
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
        _bound_frozen_review_context([{'content':'x'*(FROZEN_REVIEW_CONTEXT_MAX_CHARS+1)}])


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
    result=asyncio.run(service._run_with_agent(Attempt(session_id='blind',prompt='review'),messages=[],session_config={'frozen_review_context':True,'frozen_review_run_id':'a'*32}))
    assert result['status']=='completed'
    assert observed['persistent_memory'] is None and observed['frozen_review_context'] is True
    assert set(observed['registry']._tools)=={'mcp_trading_copilot_tc_get_daily_review_evidence'}
    assert observed['registry'].get('mcp_trading_copilot_tc_get_daily_review_evidence').is_readonly is True
    observed['event_callback']('tool_call',{'tool':'mcp_trading_copilot_tc_get_daily_review_evidence','arguments':{'source_id':'overview_1'},'call_id':'bound'})
    assert result['tool_trail'][-1]['arguments']['run_id']=='a'*32
    observed['event_callback']('tool_call',{'tool':'mcp_trading_copilot_tc_get_daily_review_evidence','arguments':{'source_id':'overview_1','run_id':'b'*32},'call_id':'conflict'})
    assert result['tool_trail'][-1]['arguments']['run_id']=='b'*32


def test_daily_tool_binds_identity_and_never_corrects_conflicting_scope():
    from src.tools.mcp import FrozenDailyReviewEvidenceTool
    from types import SimpleNamespace
    calls=[]
    remote=SimpleNamespace(name='mcp_trading_copilot_tc_get_daily_review_evidence',execute=lambda **kwargs:calls.append(kwargs) or json.dumps({'status':'ok','payload':kwargs}))
    first=FrozenDailyReviewEvidenceTool(remote,'a'*32)
    second=FrozenDailyReviewEvidenceTool(remote,'b'*32)
    assert set(first.parameters['properties'])=={'source_id'}
    assert first.parameters['required']==['source_id']
    assert first.is_readonly and first.repeatable
    assert json.loads(first.execute(source_id='overview_1'))['payload']['run_id']=='a'*32
    assert json.loads(second.execute(source_id='coverage_1'))['payload']['run_id']=='b'*32
    assert json.loads(first.execute(source_id='overview_1',run_id='b'*32))['status']=='error'
    assert json.loads(first.execute(source_id='overview_1',unsafe='value'))['status']=='error'
    assert json.loads(first.execute())['status']=='error'
    assert json.loads(first.execute(source_id=None))['status']=='error'
    assert json.loads(first.execute(source_id=''))['status']=='error'
    assert len(calls)==2
    for invalid in (None,'','a'*31,'b'*33,'not-a-run'):
        with pytest.raises(ValueError,match='run_id_required'):
            FrozenDailyReviewEvidenceTool(remote,invalid)
    with pytest.raises(ValueError,match='tool_unavailable'):
        FrozenDailyReviewEvidenceTool(SimpleNamespace(name='write_tool'),'a'*32)


def test_sole_repair_can_keep_both_complete_reads_without_trimming():
    messages=[{'role':'system','content':'x'*40000},{'role':'user','content':'x'*7000}]
    for attempt in (1,2):
        messages.append({'role':'assistant','provider_items':[{'encrypted_content':'x'*15000}]})
        messages.extend({'role':'tool','name':'mcp_trading_copilot_tc_get_daily_review_evidence','content':json.dumps({'source_id':f'page_{i}','payload':{'evidence':'x'*6500},'metadata':'x'*300}),'tool_call_id':f'{attempt}_{i}'} for i in range(62))
    original=deepcopy(messages)
    _bound_frozen_review_context(messages)
    assert messages==original


@pytest.mark.parametrize('page_count', [34, 62])
def test_agent_sees_every_prior_page_at_synthesis(monkeypatch,tmp_path:Path,page_count):
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
            if self.calls==page_count+1:
                assert len(pages)==page_count
                return SimpleNamespace(has_tool_calls=False,content='complete evidence synthesis')
            return SimpleNamespace(has_tool_calls=True,content=None,reasoning_content=None,tool_calls=[SimpleNamespace(id=str(self.calls),name='frozen_source',arguments={'page':self.calls-1})])
    spy=Spy();registry=ToolRegistry();registry.register(Source());agent=AgentLoop(registry=registry,llm=spy,frozen_review_context=True,max_iterations=page_count+6)
    run_dir=tmp_path/'run';run_dir.mkdir();agent.memory.run_dir=str(run_dir)
    result=agent.run('Read the entire day then synthesize')
    assert result['status']=='success', result
    assert spy.calls==page_count+1



def test_readonly_frozen_pages_execute_concurrently_without_losing_order(tmp_path):
    import threading
    from types import SimpleNamespace
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry, BaseTool

    gate = threading.Barrier(4)
    class Source(BaseTool):
        name = "frozen_source"
        repeatable = True
        is_readonly = True
        parameters = {"type": "object", "properties": {"page": {"type": "integer"}}, "required": ["page"]}
        def execute(self, **kwargs):
            # This fails if the known-readonly source regresses to serial reads.
            gate.wait(timeout=3)
            return json.dumps({"status": "ok", "page": kwargs["page"], "evidence": "retained"})

    class Spy:
        calls = 0
        def stream_chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(has_tool_calls=True, content=None, reasoning_content=None,
                    tool_calls=[SimpleNamespace(id=str(i), name="frozen_source", arguments={"page": i}) for i in range(4)])
            pages = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            assert pages == [{"status": "ok", "page": i, "evidence": "retained"} for i in range(4)]
            return SimpleNamespace(has_tool_calls=False, content="complete evidence synthesis")

    spy = Spy()
    registry = ToolRegistry()
    registry.register(Source())
    agent = AgentLoop(registry=registry, llm=spy, frozen_review_context=True, max_iterations=3)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    result = agent.run("Read the immutable pages")
    assert result["status"] == "success"
    assert spy.calls == 2
