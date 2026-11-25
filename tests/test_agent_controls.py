import json
from src.agent import run_agent
from src.prompts import validate_examples

class Msg:
    def __init__(self, text=None, calls=None): self.content=text; self.tool_calls=calls
    def model_dump(self, **kwargs): return {'role':'assistant','content':self.content or ''}
class Call:
    def __init__(self,name,args):
        self.id='call-1'; self.function=type('Function',(),{'name':name,'arguments':args})()
class Completion:
    def __init__(self,messages): self.messages=iter(messages); self.seen=[]
    def create(self,**kwargs):
        self.seen.append(kwargs['messages'][:]); return type('Response',(),{'choices':[type('Choice',(),{'message':next(self.messages)})()]})()
class Client:
    def __init__(self,messages):
        self.completions=Completion(messages); self.chat=type('Chat',(),{'completions':self.completions})(); self.model='test'; self.client=self

def test_agent_invalid_arguments_do_not_execute():
    client=Client([Msg(calls=[Call('search_resumes','{}')]),Msg('done')]); calls=[]
    result=run_agent(client,'q',{'search_resumes':lambda query:calls.append(query)})
    assert result['stopped']=='complete' and calls==[]
    assert 'error' in client.completions.seen[1][-1]['content']

def test_agent_iteration_limit():
    client=Client([Msg(calls=[Call('search_resumes','{"query":"x"}')])])
    assert run_agent(client,'q',{'search_resumes':lambda query:[]},max_iterations=1)['stopped']=='iteration_limit'

def test_agent_rejects_unretrieved_citation():
    body={'claims':[{'text':'claim','status':'directly_supported','citations':[{'document_id':'d','chunk_id':'c','quote':'q'}]}]}
    client=Client([Msg(json.dumps(body))])
    result=run_agent(client,'q',{},verify_final=lambda **kwargs:True)
    assert not result['source_checks_passed']

def test_few_shot_requires_reviewed_examples():
    try: validate_examples([])
    except ValueError: return
    assert False
