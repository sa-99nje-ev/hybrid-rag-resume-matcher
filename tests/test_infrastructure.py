from src.reranking import Reranker

class ExpandingTokenizer:
    def encode(self,text,text_pair=None,**kwargs):
        if text_pair is not None:return list(range(513 if len(text_pair)>200 else 100))
        return list(range(3 if text=='Q' else 500))
    def decode(self,ids,**kwargs):return 'x'*len(ids)
    def num_special_tokens_to_add(self,pair=False):return 3 if pair else 2

class FakeCrossEncoder:
    tokenizer=ExpandingTokenizer()
    def predict(self,pairs,**kwargs):
        self.pairs=pairs
        assert all(len(pair[1])<=200 for pair in pairs)
        return [1.0]*len(pairs)

def test_reranker_splits_retokenized_overflow_without_dropping_windows():
    ranker=Reranker.__new__(Reranker);ranker.model=FakeCrossEncoder()
    result=ranker.rank('Q',[('a',1)],[{'document_id':'a','text':'evidence'}])
    assert result==[('a',1.0)]
    assert sum(len(pair[1]) for pair in ranker.model.pairs)==500

from scripts.evaluate_finrank import metrics

def test_binary_metrics_use_k_denominator_and_all_gold():
    m=metrics(['g1','n','g2'],{'g1','g2'})
    assert m['P@5']==0.4
    assert m['R@5']==1.0
    assert m['MRR']==1.0
    assert abs(m['MAP']-(1+2/3)/2)<1e-12
    assert m['nDCG@5']<1.0

def test_incomplete_candidate_ranking_is_rejected():
    try:metrics(['n1','n2'],{'gold'})
    except ValueError:return
    assert False

from src.preprocessing import chunks
from src.retrieval import tokenize,bm25,rrf

def test_technical_tokens():
    assert tokenize('C++ C# .NET Node.js CI/CD') == ['c++','c#','.net','node.js','ci/cd']

def test_chunk_offsets():
    text='Financial statements\nRevenue and liabilities'
    for part in chunks('x',text,size=12,overlap=2):assert part['text']==text[part['start']:part['end']]

def test_ranking():
    assert bm25('revenue',{'a':'revenue','b':'liabilities'})[0][0]=='a'
    assert rrf([['a','b'],['b','a']])[0][0]=='a'

import io
import json
from unittest.mock import patch
from main import corpus

def test_corpus_keeps_unicode_line_separators():
    record={'id':'doc','kind':'financial','text':'first\u2028second\u0085third'}
    payload=json.dumps(record,ensure_ascii=False)+'\n'
    with patch('pathlib.Path.open',return_value=io.StringIO(payload)):
        documents, parts=corpus('unused.jsonl')
    assert documents[0]['text']==record['text']
    assert ''.join(part['text'] for part in parts)==record['text']

from decimal import Decimal
import pytest
from src.verification.numerical_verifier import decimal_expression,quotation_valid,verify

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


def test_cross_year_requires_question_periods():
    evidence={'p':'USD dollars in million for 2023 were 1,234.50', 'q':'USD dollars in million for 2022 were 1,000'}
    facts=[fact(),fact(name='b',value='1000',reporting_period='2022',passage_id='q',quotation=evidence['q'])]
    assert verify(facts,'(a-b)/b*100',evidence)['status']=='PERIOD_MISMATCH'
    result=verify(facts,'(a-b)/b*100',evidence,question='What changed between 2022 and 2023?')
    assert result['status']=='VERIFIED' and Decimal(result['result'])==Decimal('23.45')

from decimal import Decimal
import pytest
from src.verification.citation_alignment import align_quote
from src.verification.financial_quantities import canonical,compatible
from src.verification.rate_limit import TokenLimiter,estimate_tokens,duration
from src.reliability_provider import classify_provider_error
from src.verification.numerical_verifier import verify

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


def test_verifier_converts_scales_and_rejects_unit_combinations():
    a={'name':'a','value':'1','currency':'USD','unit':'dollars','magnitude':'billion','reporting_period':'2023','passage_id':'p','quotation':'USD dollars in billion during 2023 were 1'}
    b={'name':'b','value':'500','currency':'USD','unit':'dollars','magnitude':'million','reporting_period':'2023','passage_id':'q','quotation':'USD dollars in million during 2023 were 500'}
    evidence={'p':a['quotation'],'q':b['quotation']}
    assert Decimal(verify([a,b],'a+b',evidence)['result'])==Decimal('1.5')
    assert verify([a,b],'a*b',evidence)['status']=='UNIT_MISMATCH'
    assert Decimal(verify([a,b],'(a-b)/b*100',evidence)['result'])==Decimal('100')


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
