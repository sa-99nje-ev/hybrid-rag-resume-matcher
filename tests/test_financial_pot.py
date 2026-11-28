import copy
import pytest
from decimal import Decimal
from pydantic import ValidationError
from src.financial_pot import FinancialPlan,finalize,answer_with_context

Q='What was revenue percentage change from 2022 to 2023?'
C=[{'passage_id':'p1','text':'USD dollars in million for 2022 revenue were 100'},
   {'passage_id':'p2','text':'USD dollars in million for 2023 revenue were 120'}]
def plan():
 return {'evidence_plan':{'required_line_items':['revenue'],'required_periods':['2022','2023'],'required_comparisons':['change'],'operation_explanation':'Compare periods.'},
 'extracted_facts':[{'name':'old','line_item':'revenue','value':'100','currency':'USD','unit':'dollars','magnitude':'million','reporting_period':'2022','passage_id':'p1','quotation':C[0]['text']},
                    {'name':'new','line_item':'revenue','value':'120','currency':'USD','unit':'dollars','magnitude':'million','reporting_period':'2023','passage_id':'p2','quotation':C[1]['text']}],
 'calculation':{'expression':'(new-old)/old*100','input_variables':['old','new'],'expected_output_unit':'percent','calculation_type':'percentage_change','baseline_variable':'old','current_variable':'new','numerator_variable':'','denominator_variable':''},
 'citations':[{'passage_id':'p1','quotation':C[0]['text']},{'passage_id':'p2','quotation':C[1]['text']}],
 'response_intent':{'answer_type':'numerical','explanation':'Compare revenue.','numerical_calculation_required':True},
 'abstention':{'insufficient_evidence':False,'reason':''}}

def test_pot_percentage_assembled_locally():
 r=finalize(Q,C,plan())
 assert r['status']=='calculation_checked' and r['answer']=='20.0 %'
 assert r['operation']=='(new-old)/old*100' and r['semantic_support']=='not automatically verified'
 assert r['citations'][0]['original_quote']==C[0]['text'] and r['citations'][0]['exact_span']==C[0]['text']

def test_missing_required_field_and_invalid_decimal():
 p=plan();del p['evidence_plan'];assert finalize(Q,C,p)['reason']=='SCHEMA_INVALID'
 p=plan();p['extracted_facts'][0]['value']='1e3';assert finalize(Q,C,p)['reason']=='SCHEMA_INVALID'

def test_ratio():
 p=plan();p['evidence_plan']['required_periods']=['2022'];p['evidence_plan']['required_line_items']=['revenue','liabilities']
 p['extracted_facts'][1].update(line_item='liabilities',value='50',reporting_period='2022',passage_id='p3',quotation='USD dollars in million for 2022 liabilities were 50')
 p['citations'][1]={'passage_id':'p3','quotation':p['extracted_facts'][1]['quotation']}
 p['calculation'].update(expression='old/new',expected_output_unit='ratio',calculation_type='ratio',numerator_variable='old',denominator_variable='new')
 r=finalize('What was the ratio in 2022?',C+[{'passage_id':'p3','text':p['extracted_facts'][1]['quotation']}],p)
 assert r['status']=='calculation_checked' and r['answer']=='2 ratio'

def test_scale_conversion():
 p=plan();p['extracted_facts'][1].update(value='0.12',magnitude='billion',quotation='USD dollars in billion for 2023 revenue were 0.12')
 c=[C[0],{'passage_id':'p2','text':p['extracted_facts'][1]['quotation']}];p['citations'][1]['quotation']=c[1]['text']
 assert Decimal(finalize(Q,c,p)['answer'].split()[0])==Decimal('20')

@pytest.mark.parametrize('change,reason',[
 (lambda p:p['extracted_facts'][1].update(currency='EUR'),'UNIT_MISMATCH'),
 (lambda p:p['extracted_facts'][1].update(reporting_period='2021'),'PERIOD_MISMATCH'),
 (lambda p:p['calculation'].update(expression='__import__("os")'),'INVALID_VARIABLES'),
 (lambda p:p['calculation'].update(expression='(old-new)/old*100'),'INVALID_COMPARISON'),
 (lambda p:p['extracted_facts'][1].update(value='999'),'SOURCE_VALUE_UNVERIFIED'),
 (lambda p:p['citations'][0].update(passage_id='missing'),'INVALID_CITATION'),
 (lambda p:p['extracted_facts'][0].update(line_item='profit'),'LINE_ITEM_UNVERIFIED'),
 ])
def test_pot_rejects_unsupported(change,reason):
 p=plan();change(p);assert finalize(Q,C,p)['reason']==reason

def test_ambiguous_citation_abstains():
 p=plan();context=[{'passage_id':'p1','text':C[0]['text']+' '+C[0]['text']},C[1]]
 assert finalize(Q,context,p)['reason']=='INVALID_CITATION'

def test_model_abstention():
 p=plan();p['abstention']={'insufficient_evidence':True,'reason':'missing'}
 assert finalize(Q,C,p)['reason']=='MODEL_ABSTENTION'

def test_qualitative_does_not_claim_entailment():
 p=plan();p['response_intent']={'answer_type':'qualitative','explanation':'a claim','numerical_calculation_required':False}
 result=finalize(Q,C,p);assert result['status']=='abstained' and result['reason']=='QUALITATIVE_REVIEW_REQUIRED' and result['answer'] is None

def test_single_call_and_provider_schema():
 class Mock:
  def __init__(self):self.calls=[]
  def request(self,question,context,**kwargs):
   self.calls.append(kwargs);return FinancialPlan.model_validate(plan()),'{"mock":true}',{'status':'ok'}
 provider=Mock();result=answer_with_context(Q,C,provider)
 assert result['status']=='calculation_checked' and len(provider.calls)==1
 assert provider.calls[0]['max_attempts']==1 and provider.calls[0]['schema_type'] is FinancialPlan

def test_malformed_provider_response_abstains():
 class Mock:
  def request(self,*args,**kwargs):return None,'{bad',{'status':'schema_invalid'}
 assert answer_with_context(Q,C,Mock())['reason']=='SCHEMA_INVALID'

def test_pipeline_retrieval_reranks_and_calls_once(monkeypatch):
 import src.financial_pipeline as p
 monkeypatch.setattr(p,'rank',lambda *a,**k:[('d1',1),('d2',.5)])
 class Rerank:
  def rank(self,*a,**k):return [('d2',1),('d1',.5)]
 class Provider:
  def __init__(self):self.count=0
  def request(self,*args,**kwargs):self.count+=1;return FinancialPlan.model_validate(plan()),'{}',{'status':'ok'}
 parts=[{'document_id':'d1','chunk_id':'p1','text':C[0]['text'],'start':0},{'document_id':'d2','chunk_id':'p2','text':C[1]['text'],'start':0}]
 provider=Provider();result=p.answer(Q,[],parts,None,None,Rerank(),provider)
 assert result['status']=='calculation_checked' and provider.count==1 and result['context_passage_ids']==['p2','p1']


def test_ambiguous_figure_attribution_abstains():
 p=plan();ambiguous='USD dollars in million for 2022 revenue 100 and liabilities 200'
 context=[{'passage_id':'p1','text':ambiguous},C[1]]
 p['extracted_facts'][0]['value']='200';p['extracted_facts'][0]['quotation']=ambiguous;p['citations'][0]['quotation']=ambiguous
 assert finalize(Q,context,p)['reason']=='SOURCE_VALUE_AMBIGUOUS'
