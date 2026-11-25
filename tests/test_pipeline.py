import json
from src.agent import run_agent
from src.generation import answer
from src.llm import Report

class FakeResponse:
    def __init__(self, message): self.choices=[type('Choice',(),{'message':message})()]
class FakeMessage:
    tool_calls=None
    content='done'
    def model_dump(self, **kwargs): return {'role':'assistant','content':self.content}
class FakeLLM:
    model='test'
    def __init__(self):
        self.client=type('Client',(),{'chat':type('Chat',(),{'completions':type('Completions',(),{'create':lambda self,**kwargs:FakeResponse(FakeMessage())})()})()})()
    def generate(self, instruction, data, schema):
        return Report.model_validate({'claims':[{'text':'example','status':'directly_supported','citations':[{'document_id':'a','chunk_id':'c','quote':'C++'}]}]}), {'latency_seconds':0}

def test_grounding_rejects_fabricated_quote():
    doc=[{'id':'a','text':'C++'}]
    chunk=[{'document_id':'a','chunk_id':'c','text':'C++','start':0,'end':3}]
    result=answer('q',[('a',1)],doc,chunk,FakeLLM())
    assert result['source_checks_passed']
    chunk[0]['text']='C#'
    assert not answer('q',[('a',1)],doc,chunk,FakeLLM())['source_checks_passed']

def test_agent_stops_when_model_answers():
    assert run_agent(FakeLLM(),'q',{},max_iterations=2)['stopped']=='complete'
