from decimal import Decimal
import pytest
from pydantic import ValidationError
from src.verification.numerical_verifier import decimal_expression,normalize,quotation_valid,verify
from src.financial_milestone2 import Response,citation_status,gate,classify,retry_delay

@pytest.mark.parametrize('expr,expected', [('(a-b)/b*100','25'),('a/b','1.25'),('0.1+0.2','0.3'),('-a + b','-2')])
def test_decimal_arithmetic(expr,expected):
    values={'a':'10','b':'8'}
    assert decimal_expression(expr,values)==Decimal(expected)

@pytest.mark.parametrize('expr', ['__import__("os")','a.__class__','a**2','a[0]','abs(a)','a//b','a%2','unknown'])
def test_reject_unsafe_expression(expr):
    with pytest.raises((ValueError,SyntaxError)):
        decimal_expression(expr,{'a':'2','b':'1'})

def test_quote_normalization():
    assert quotation_valid('x','caf\u00e9  income',{'x':'cafe\u0301 \n income'})
    assert not quotation_valid('y','income',{'x':'income'})

def fact(**kw):
    value={'name':'a','value':'1234.50','currency':'USD','unit':'dollars','magnitude':'million','reporting_period':'2023','passage_id':'p','quotation':'USD dollars in million for 2023 were 1,234.50'}
    value.update(kw);return value

def test_source_grounding_and_missing_number():
    evidence={'p':'USD dollars in million for 2023 were 1,234.50'}
    assert verify([fact()],'a/2',evidence)['result']=='617.25'
    assert verify([fact(value='1235')],'a/2',evidence)['status']=='SOURCE_VALUE_UNVERIFIED'
    assert verify([fact(reporting_period='2022')],'a/2',evidence)['status']=='PERIOD_MISMATCH'
    assert verify([fact(unit='shares')],'a/2',evidence)['status']=='UNIT_MISMATCH'
    assert verify([fact()],'a/0',evidence)['status']=='INVALID_FORMULA'
    assert verify([],'',evidence)['status']=='INSUFFICIENT_EVIDENCE'

def test_malformed_json():
    with pytest.raises(ValidationError):Response.model_validate_json('{bad}')
    with pytest.raises(ValidationError):Response.model_validate_json('{}')

def test_gate_missing_citation_and_numerical():
    report=Response(answer='A',abstention=False,abstention_reason='',extracted_facts=[],formula='',citations=[],verification_status='unverified')
    status=citation_status(report,{})
    assert gate(report,status,{'status':'INSUFFICIENT_EVIDENCE'})['abstain']

def test_retry_classification_and_delay():
    class ResponseStub:headers={'retry-after':'8'}
    class Failure(Exception):
        status_code=429
        response=ResponseStub()
    err=Failure()
    assert classify(err)=='rate_limit' and retry_delay(err,0)==8
    assert classify(ValueError())=='non_retryable'


def test_retry_is_bounded(monkeypatch):
    import src.financial_milestone2 as m
    class RateLimit(Exception):
        status_code=429
        response=type('R',(),{'headers':{'retry-after':'0'}})()
    class Create:
        calls=0
        def create(self,**kwargs):
            self.calls+=1
            raise RateLimit()
    request=Create();provider=m.Provider.__new__(m.Provider);provider.model='mock';provider.client=type('C',(),{'chat':type('Chat',(),{'completions':request})()})()
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    obj,raw,stats=provider.request('q',[],max_attempts=3)
    assert obj is None and raw is None
    assert stats['status']=='provider_error' and stats['attempts']==3 and request.calls==3
    assert stats['events']==['rate_limit']*3

def test_id_and_quote_checks_separate():
    report=Response.model_validate({'answer':'answer','abstention':False,'abstention_reason':'','extracted_facts':[],'formula':'','citations':[{'passage_id':'p','quotation':'fabricated'}],'verification_status':'unverified'})
    assert citation_status(report,{'p':'source'})=={'passage_ids_valid':True,'quotations_valid':False}


def test_provider_malformed_response_records_schema_failure():
    import src.financial_milestone2 as m
    from types import SimpleNamespace
    result=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{bad'))],usage=None,id='request')
    request=SimpleNamespace(create=lambda **kwargs:result)
    provider=m.Provider.__new__(m.Provider);provider.model='mock';provider.client=SimpleNamespace(chat=SimpleNamespace(completions=request))
    obj,raw,stats=provider.request('q',[])
    assert obj is None and raw=='{bad' and stats['status']=='schema_invalid'

def test_summary_quantile():
    from scripts.summarize_finrank_generation import quantile
    assert quantile([],0.5) is None
    assert quantile([0,2],0.5)==1



def test_cross_year_requires_question_periods():
    evidence={'p':'USD dollars in million for 2023 were 1,234.50', 'q':'USD dollars in million for 2022 were 1,000'}
    facts=[fact(),fact(name='b',value='1000',reporting_period='2022',passage_id='q',quotation=evidence['q'])]
    assert verify(facts,'(a-b)/b*100',evidence)['status']=='PERIOD_MISMATCH'
    result=verify(facts,'(a-b)/b*100',evidence,question='What changed between 2022 and 2023?')
    assert result['status']=='VERIFIED' and Decimal(result['result'])==Decimal('23.45')

def test_gate_checks_computed_answer():
    report=Response(answer='2',abstention=False,abstention_reason='',extracted_facts=[fact()],formula='a/2',citations=[{'passage_id':'p','quotation':'source'}],verification_status='unverified')
    decision=gate(report,{'quotations_valid':True},{'status':'VERIFIED','result':'5'})
    assert 'answer_does_not_match_calculation' in decision['reasons']


def test_checkpoint_deduplicates_success_and_retries_failure(tmp_path):
    import json
    from scripts.eval_finrank_generation import completed_keys
    path=tmp_path/'pred.jsonl'
    entries=[{'query_id':'q','method':'a','status':'ok'},{'query_id':'q','method':'a','status':'ok'},{'query_id':'q','method':'b','status':'provider_error'}]
    path.write_text('\n'.join(json.dumps(x) for x in entries),encoding='utf8')
    assert completed_keys(path)=={('q','a')}
