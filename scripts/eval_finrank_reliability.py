"""Local-first five-question dev reliability experiment; held-out split unavailable."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT.parent/'.env',override=False)
from src.finrank import load_finrank
from src.financial_milestone2 import citation_status,gate,Response
from src.verification.citation_alignment import align_quote
from src.verification.numerical_verifier import verify
from src.verification.financial_quantities import answer_agreement
from src.verification.rate_limit import estimate_tokens
from src.reliability_provider import ReliabilityProvider,PROMPT
def append(path,item):
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def completed_keys(path):
    completed=set()
    if path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            if row['status'] not in ('provider_error','provider_deferred'):
                completed.add((row['query_id'],row['method']))
    return completed
METHODS=('G1_BM25','G2_Hybrid','G3_Hybrid_CE');TOP_K=8;MAX_CHARS=15000

def context_for(ranking,lookup,max_chars=MAX_CHARS,max_passages=TOP_K):
    selected=[];omitted=[];chars=0
    for pid in ranking[:max_passages]:
        text=lookup[pid]
        if chars+len(text)>max_chars:omitted.append({'passage_id':pid,'reason':'full_passage_over_budget'});continue
        selected.append({'passage_id':pid,'text':text});chars+=len(text)
    return selected,omitted

def prior_context(ranking,lookup):
    selected=[];chars=0
    for pid in ranking:
        if len(selected)>=5:break
        text=lookup[pid]
        if chars+len(text)>16000:continue
        selected.append({'passage_id':pid,'text':text});chars+=len(text)
    return selected

def local_align(report,evidence):
    data=report.model_dump();audit=[]
    for collection in ('citations','extracted_facts'):
        for item in data[collection]:
            match=align_quote(item['passage_id'],item['quotation'],evidence);audit.append({'collection':collection,**match})
            if match['status'] in ('exact','aligned_punctuation_only'):item['quotation']=match['exact_span']
    return Response.model_validate(data),audit

def run(limit,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);data=ROOT/'data/raw/finrank';records,docs,aliases,_,_=load_finrank(data)
    byid={r['id']:r for r in records};lookup={d['passage_id']:d['passage_text'] for d in docs}
    saved=ROOT/'results/milestone2_dev/dev_retrieval.jsonl';rankings=[json.loads(x) for x in saved.read_text(encoding='utf8').splitlines()][:limit]
    if len(rankings)!=limit:raise ValueError('Only existing dev rankings may be used')
    code=[Path(__file__),ROOT/'src/financial_milestone2.py',ROOT/'src/reliability_provider.py',ROOT/'src/verification/numerical_verifier.py',ROOT/'src/verification/financial_quantities.py',ROOT/'src/verification/citation_alignment.py',ROOT/'src/verification/rate_limit.py']
    cfg={'split':'dev','query_ids':[x['query_id'] for x in rankings],'regime':'saved global pool of 5230 passages','retrieval_sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),
      'dataset_sha256':hashlib.sha256((data/'FinRank.jsonl').read_bytes()).hexdigest(),'model':os.getenv('GROQ_MODEL','openai/gpt-oss-20b'),
      'top_k':TOP_K,'max_context_characters':MAX_CHARS,'context_policy':'entire passages only, omit and log passages exceeding budget',
      'temperature':0,'max_completion_tokens':1024,'citation_repair':'local deterministic alignment; no provider calls',
      'answer_tolerance':'max(0.01, 0.1% of computed value); one numeric token only',
      'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code}}
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8'))['signature']!=signature:raise RuntimeError('Run signature mismatch; choose a fresh directory')
    else:manifest.write_text(json.dumps({'signature':signature,'configuration':cfg},indent=2),encoding='utf8')
    pred=out/'predictions.jsonl';rawfile=out/'raw_diagnostics.jsonl';done=completed_keys(pred);provider=ReliabilityProvider(cfg['model'],os.environ['GROQ_API_KEY'])
    for ranking in rankings:
        q=byid[ranking['query_id']];gold={aliases[x['passage_id']] for x in q['passages']}
        for method in METHODS:
            if (q['id'],method) in done:continue
            context,omitted=context_for(ranking[method],lookup);evidence={x['passage_id']:x['text'] for x in context}
            old=prior_context(ranking[method],lookup)
            def estimate(ctx):return estimate_tokens(PROMPT+json.dumps(Response.model_json_schema())+json.dumps({'question':q['question'],'evidence':ctx},ensure_ascii=False))
            row={'query_id':q['id'],'method':method,'regime':'saved_global_dev','context_ids':list(evidence),'omitted':omitted,
                 'known_gold_context_recall':len(set(evidence)&gold)/len(gold),'new_prompt_estimate':estimate(context),'prior_context_prompt_estimate':estimate(old)}
            if not context:row.update(status='no_context')
            else:
                report,raw,stats=provider.request(q['question'],context);row['provider']=stats
                append(rawfile,{'query_id':q['id'],'method':method,'request_id':stats['request_id'],'raw_response':raw})
                if report is None:row['status']=stats['status']
                else:
                    first=citation_status(report,evidence);aligned,audit=local_align(report,evidence);final=citation_status(aligned,evidence)
                    numerical=verify([x.model_dump() for x in aligned.extracted_facts],aligned.formula,evidence,question=q['question'])
                    agreement=answer_agreement(aligned.answer,numerical['result']) if numerical['status']=='VERIFIED' else False
                    numerical['answer_status']='MATCH' if agreement else 'NOT_VERIFIED';decision=gate(aligned,final,numerical)
                    row.update(status='ok',first_citations=first,final_citations=final,citation_audit=audit,numerical=numerical,gate=decision,
                      model_abstention=aligned.abstention,numerical_candidate=bool(aligned.formula or aligned.extracted_facts),
                      verified_answer=None if decision['abstain'] else aligned.answer)
            append(pred,row);print('Recorded',method,row['status'],flush=True)
            if row['status']=='provider_deferred' or 'rate_limit' in row.get('provider',{}).get('events',[]):
                print('Stopping batch for provider Retry-After; resume after cooldown',flush=True);return
    print('Reliability dev run complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev'],default='dev');p.add_argument('--limit',type=int,default=5);p.add_argument('--out',default='results/milestone2_reliability_dev/run5');a=p.parse_args()
    if not 1<=a.limit<=10:p.error('only 1 to 10 existing dev rankings supported')
    run(a.limit,a.out)
