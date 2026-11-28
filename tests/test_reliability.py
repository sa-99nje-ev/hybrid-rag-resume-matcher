from decimal import Decimal
import pytest
from src.verification.citation_alignment import align_quote
from src.verification.financial_quantities import canonical,compatible,answer_agreement
from src.verification.rate_limit import TokenLimiter,estimate_tokens,duration
from src.reliability_provider import classify_provider_error
from src.verification.numerical_verifier import verify
from scripts.eval_finrank_reliability import context_for,local_align
from src.financial_milestone2 import Response,gate

def test_exact_original_span_and_unicode():
    source='The cafe\u0301 had revenue of $4 million. Other.'
    x=align_quote('p','café had revenue of $4 million',{'p':source})
    assert x['status']=='exact' and x['exact_span']=='cafe\u0301 had revenue of $4 million'

def test_fuzzy_unique_and_ambiguous():
    quote='Total assets were 10 million dollars, and remain unchanged'
    src='Total assets were 10 million dollars and remain unchanged'
    assert align_quote('p',quote,{'p':src})['status']=='aligned_punctuation_only'
    assert align_quote('p',quote,{'p':src+'; '+src})['status']=='ambiguous'

def test_fuzzy_reject_numerical_negation_and_dates():
    src='It did not have 20 million dollars in fiscal 2022.'
    for quote in ['It did not have 30 million dollars in fiscal 2022','It did have 20 million dollars in fiscal 2022','It did not have 20 million dollars in fiscal 2023']:
        assert align_quote('p',quote,{'p':src})['status']=='invalid'

def test_canonical_values_magnitudes_and_dimensions():
    a=canonical({'value':'1.5','magnitude':'billion','unit':'dollars','currency':'USD'})
    b=canonical({'value':'1500','magnitude':'million','unit':'dollars','currency':'USD'})
    assert Decimal(a['canonical_value'])==Decimal(b['canonical_value']) and compatible(a,b)
    pct=canonical({'value':'5','magnitude':'','unit':'percent','currency':''})
    assert not compatible(a,pct)
    with pytest.raises(ValueError):canonical({'value':'5','magnitude':'million','unit':'percent','currency':''})

def test_answer_agreement_tolerance_and_ambiguity():
    assert answer_agreement('23.46%',Decimal('23.45'))
    assert not answer_agreement('24%',Decimal('23.45'))
    assert not answer_agreement('in 2023, 23.45%',Decimal('23.45'))

def test_context_budget_preserves_passages_and_reports_omissions():
    selected,omitted=context_for(['a','b','c'],{'a':'12345','b':'12345','c':'1'},max_chars=6,max_passages=3)
    assert [x['passage_id'] for x in selected]==['a','c'] and omitted==[{'passage_id':'b','reason':'full_passage_over_budget'}]

def test_limiter_limits_and_estimate(monkeypatch,tmp_path):
    monkeypatch.setenv('GROQ_TPM','100');monkeypatch.setenv('GROQ_RPM','10');monkeypatch.setenv('GROQ_RPD','3')
    class Timer:
        now=0
        def clock(self):return self.now
        def sleep(self,n):self.now+=n
    clock=Timer();l=TokenLimiter(clock=clock.clock,sleep=clock.sleep);l.cooldown_file=tmp_path/'cooldown.json'
    l.wait(40,10);l.wait(40,10);l.wait(40,10)
    assert clock.now>=60 and estimate_tokens('abcdef')==2 and duration('1m2.5s')==62.5
    with pytest.raises(ValueError):l.wait(1,1)

def test_http_400_classification():
    class E(Exception):status_code=400;body={'message':'invalid json_schema'}
    assert classify_provider_error(E())=='bad_request_schema'
    E.body={'message':'maximum context length exceeded'};assert classify_provider_error(E())=='bad_request_context'
    E.body={'message':'unknown'};assert classify_provider_error(E())=='bad_request_other'

def test_local_alignment_does_not_invoke_provider():
    r=Response.model_validate({'answer':'x','abstention':False,'abstention_reason':'','extracted_facts':[],'formula':'','citations':[{'passage_id':'p','quotation':'the total is 100 dollars, for the year'}],'verification_status':'pending'})
    updated,audit=local_align(r,{'p':'the total is 100 dollars for the year'})
    assert audit[0]['status']=='aligned_punctuation_only'
    assert updated.citations[0].quotation=='the total is 100 dollars for the year'

def test_abstention_not_relaxed_for_invalid_evidence():
    r=Response.model_validate({'answer':'4','abstention':False,'abstention_reason':'','extracted_facts':[],'formula':'','citations':[],'verification_status':'pending'})
    assert gate(r,{'quotations_valid':False},{'status':'VERIFIED','result':'4'})['abstain']


def test_verifier_converts_scales_and_rejects_unit_combinations():
    a={'name':'a','value':'1','currency':'USD','unit':'dollars','magnitude':'billion','reporting_period':'2023','passage_id':'p','quotation':'USD dollars in billion during 2023 were 1'}
    b={'name':'b','value':'500','currency':'USD','unit':'dollars','magnitude':'million','reporting_period':'2023','passage_id':'q','quotation':'USD dollars in million during 2023 were 500'}
    evidence={'p':a['quotation'],'q':b['quotation']}
    assert Decimal(verify([a,b],'a+b',evidence)['result'])==Decimal('1.5')
    assert verify([a,b],'a*b',evidence)['status']=='UNIT_MISMATCH'
    assert Decimal(verify([a,b],'(a-b)/b*100',evidence)['result'])==Decimal('100')


def test_prior_budget_scans_beyond_five_candidates():
    from scripts.eval_finrank_reliability import prior_context
    rankings=['a','b','c','d','e','f']
    selected=prior_context(rankings,{'a':'x'*17000,'b':'1','c':'2','d':'3','e':'4','f':'5'})
    assert [x['passage_id'] for x in selected]==['b','c','d','e','f']


def test_cooldown_persists_and_blocks_future_requests(tmp_path):
    from src.verification.rate_limit import ProviderDeferred
    a=TokenLimiter();a.cooldown_file=tmp_path/'cooldown.json';a.set_cooldown(60)
    b=TokenLimiter();b.cooldown_file=a.cooldown_file
    with pytest.raises(ProviderDeferred):b.wait(10,10)

def test_percentage_source_and_invalid_formula():
    x={'name':'a','value':'5','currency':'','unit':'percent','magnitude':'','reporting_period':'2023','passage_id':'p','quotation':'the percent rate in 2023 was 5'}
    assert verify([x],'a/100',{'p':x['quotation']})['status']=='VERIFIED'
    assert verify([x],'__import__("os")',{'p':x['quotation']})['status']=='INVALID_FORMULA'


def test_overlapping_exact_matches_are_ambiguous():
    assert align_quote('p','aaa',{'p':'aaaa'})['status']=='ambiguous'

def test_reported_token_remaining_is_respected(monkeypatch,tmp_path):
    class Timer:
        now=0
        def clock(self):return self.now
        def sleep(self,n):self.now+=n
    clock=Timer();limiter=TokenLimiter(clock=clock.clock,sleep=clock.sleep);limiter.cooldown_file=tmp_path/'none'
    limiter.observe({'x-ratelimit-remaining-tokens':'50','x-ratelimit-reset-tokens':'7s'})
    limiter.wait(40,20)
    assert clock.now>=7


def test_deferred_or_failed_request_can_resume(tmp_path):
    import json
    from scripts.eval_finrank_reliability import completed_keys
    path=tmp_path/'predictions.jsonl'
    rows=[{'query_id':'q','method':'m','status':'provider_deferred'},{'query_id':'q','method':'n','status':'ok'}]
    path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
    assert completed_keys(path)=={('q','n')}
