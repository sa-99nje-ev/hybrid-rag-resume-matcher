from src.financial_rag import FinancialResponse,financial_answer,validate_financial_response

def response(quote,abstain=False):
    return FinancialResponse(question='Question?',answer='Answer',company=None,reporting_period=None,
        numerical_values=[],explanation='Explanation',citations=[{'passage_id':'p1','quotation':quote}] if quote else [],
        insufficient_evidence=abstain,abstention_reason='Insufficient' if abstain else None)

class FakeLLM:
    def __init__(self,reports):self.reports=iter(reports);self.calls=0
    def generate(self,instruction,data,schema):
        self.calls+=1
        return next(self.reports),{'latency_seconds':1.0,'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}

def test_citation_correction_is_bounded_and_tracks_usage():
    model=FakeLLM([response('not a quote'),response('source evidence')])
    report,stats=financial_answer('Question?',[{'passage_id':'p1','text':'source evidence'}],model)
    assert stats['source_reference_valid'] and model.calls==2
    assert stats['first_pass_errors'] and stats['usage']['total_tokens']==10
    assert report['citations'][0]['quotation']=='source evidence'

def test_repeated_invalid_citations_remain_invalid():
    model=FakeLLM([response('bad'),response('bad')])
    _,stats=financial_answer('Question?',[{'passage_id':'p1','text':'source evidence'}],model)
    assert not stats['source_reference_valid'] and model.calls==2

def test_abstention_requires_reason():
    item=response('',True);item.abstention_reason=None
    assert 'abstention reason missing' in validate_financial_response('Question?',[],item)
