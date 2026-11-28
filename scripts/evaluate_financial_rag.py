"""Run a controlled non-commercial FinRank RAG pilot; reference answers stay local."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT.parent/'.env',override=False)
os.environ.setdefault('HF_HOME',str(ROOT.parent/'advanced-resume-rag/data/model-cache'))
import numpy as np
from rank_bm25 import BM25Okapi
from groq import RateLimitError
from src.finrank import load_finrank
from src.embeddings import Encoder
from src.vector_store import Store
from src.retrieval import tokenize,rrf
from src.reranking import Reranker
from src.llm import LLM
from src.financial_rag import financial_answer

DEPTH=20; MAX_PASSAGES=5; MAX_CONTEXT_CHARS=16000

def context_for(ranking,lookup):
    context=[];count=0; omitted=[]
    for pid in ranking:
        text=lookup[pid]
        if len(context)>=MAX_PASSAGES:break
        if count+len(text)>MAX_CONTEXT_CHARS:omitted.append(pid);continue
        context.append({'passage_id':pid,'text':text});count+=len(text)
    return context,omitted

def call_with_rate_limit_retry(question,context,llm):
    for attempt in range(2):
        try:return financial_answer(question,context,llm),attempt
        except RateLimitError as exc:
            if attempt:raise
            try:delay=float(exc.response.headers.get('retry-after','5'))
            except (AttributeError,TypeError,ValueError):delay=5.0
            if not 0<=delay<=45:raise
            time.sleep(delay+0.1)
def run(split,limit,out):
    data=ROOT/'data/raw/finrank'; records,docs,aliases,test,audit=load_finrank(data)
    splits=json.loads((data/'baselines/splits.json').read_text(encoding='utf-8'))['random']
    permitted=set(splits[split]);byid={r['id']:r for r in records}
    queries=sorted((byid[i] for i in permitted),key=lambda r:r['id'])
    if split=='test':
        allowed={r['id'] for r in test};queries=[r for r in queries if r['id'] in allowed]
    if limit is not None:queries=queries[:limit]
    if not queries:raise ValueError('No questions selected')
    store=Store(collection='finrank_passages'); info=store.client.get_collection(store.collection)
    if info.points_count!=len(docs) or info.indexed_vectors_count!=len(docs) or info.status.value!='green':raise RuntimeError('FinRank collection not ready')
    index_file=ROOT/'data/processed/finrank_index_signature.json'; corpus_file=ROOT/'data/processed/finrank_passages.jsonl'
    if not index_file.exists():raise RuntimeError('FinRank index signature is missing')
    index_signature=json.loads(index_file.read_text(encoding='utf-8'))
    if index_signature.get('corpus_sha256')!=hashlib.sha256(corpus_file.read_bytes()).hexdigest():raise RuntimeError('Index signature does not match prepared corpus')
    docids=[d['passage_id'] for d in docs];lookup={d['passage_id']:d['passage_text'] for d in docs}
    bm=BM25Okapi([tokenize(d['passage_text']) for d in docs]);encoder=Encoder(cache_dir=str(ROOT/'data/cache/finrank_embeddings'),batch_size=16)
    reranker=Reranker();llm=LLM();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'finrank_rag_predictions.jsonl'; manifest_path=out/'finrank_rag_manifest.json'
    config={'split':split,'limit':limit,'query_order':'sorted record ID','query_ids':[r['id'] for r in queries],
            'model':llm.model,'temperature':0,'retrieval_depth':DEPTH,'max_passages':MAX_PASSAGES,'max_context_chars':MAX_CONTEXT_CHARS,
            'dataset_sha256':hashlib.sha256((data/'FinRank.jsonl').read_bytes()).hexdigest(),
            'processed_corpus_sha256':index_signature['corpus_sha256'],'qdrant_collection':'finrank_passages',
            'embedding_model':index_signature['embedding_model'],
            'model_revisions':{m:(Path(os.environ['HF_HOME'])/'hub'/('models--'+m.replace('/','--'))/'refs/main').read_text().strip() for m in ['BAAI/bge-small-en-v1.5','BAAI/bge-reranker-base']},
            'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'src/financial_rag.py',ROOT/'src/llm.py',ROOT/'src/retrieval.py',ROOT/'src/reranking.py',ROOT/'src/embeddings.py']},
            'source_license':'CC BY-NC 4.0, FinRank (Mansouri et al.)','source_url':'https://github.com/datanxt/FinRank',
            'note':'Global retrieval only, no oracle gold or query rewrites passed to models. Character budgeting omits entire passages; it is not a token count.'}
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('signature')!=signature:raise RuntimeError('Output directory has a different configuration; choose another directory')
    else:manifest_path.write_text(json.dumps({'signature':signature,'configuration':config},indent=2),encoding='utf-8')
    complete=set()
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item=json.loads(line);complete.add((item['query_id'],item['method']))
    with path.open('a',encoding='utf-8',newline='\n') as target:
        for row in queries:
            needed=[x for x in ('G1_Naive','G2_Hybrid','G3_Advanced') if (row['id'],x) not in complete]
            if not needed:continue
            q=row['question']; gold={aliases[p['passage_id']] for p in row['passages']}
            t=time.perf_counter();vector=encoder.encode([q],query=True)[0];dense=[p.payload['document_id'] for p in store.search(vector,limit=DEPTH)];dense_time=time.perf_counter()-t
            t=time.perf_counter();scores=np.asarray(bm.get_scores(tokenize(q)));lex=[docids[i] for i in np.argsort(-scores,kind='stable')[:DEPTH]];bm_time=time.perf_counter()-t
            t=time.perf_counter();hybrid=[d for d,_ in rrf([lex,dense])[:DEPTH]];fusion_time=time.perf_counter()-t
            t=time.perf_counter();chunks=[{'document_id':pid,'text':lookup[pid]} for pid in hybrid];advanced=[pid for pid,_ in reranker.rank(q,[(x,0.0) for x in hybrid],chunks,depth=DEPTH)];rerank_time=time.perf_counter()-t
            configs={'G1_Naive':(dense,dense_time),'G2_Hybrid':(hybrid,dense_time+bm_time+fusion_time),'G3_Advanced':(advanced,dense_time+bm_time+fusion_time+rerank_time)}
            for method in needed:
                ranking,retrieval_time=configs[method];context_started=time.perf_counter();context,omitted=context_for(ranking,lookup);context_time=time.perf_counter()-context_started
                result={'query_id':row['id'],'method':method,'reasoning_type':row.get('reasoning_type'),
                        'retrieved_ids':ranking,'context_ids':[c['passage_id'] for c in context],
                        'omitted_for_budget':omitted,'retrieval_seconds':retrieval_time}
                if not context:
                    result.update({'status':'no_context','schema_valid':False})
                else:
                    try:
                        call_started=time.perf_counter()
                        (response,stats),rate_retries=call_with_rate_limit_retry(q,context,llm)
                        api_wall_seconds=time.perf_counter()-call_started
                        evidence={item['passage_id']:item['text'] for item in context}
                        cited={x['passage_id'] for x in response['citations'] if x['passage_id'] in evidence and x['quotation'].strip() and x['quotation'] in evidence[x['passage_id']]}
                        result.update({'status':'ok','schema_valid':True,'response':response,'stats':stats,
                                       'known_gold_citation_coverage':len(cited&gold)/len(gold),
                                       'known_gold_context_coverage':len(set(result['context_ids'])&gold)/len(gold),
                                       'retrieval_plus_api_seconds':retrieval_time+context_time+api_wall_seconds,
                                       'api_wall_seconds':api_wall_seconds,'context_seconds':context_time,'rate_limit_retries':rate_retries})
                    except Exception as exc:
                        result.update({'status':'api_or_validation_error','schema_valid':False,'error_type':type(exc).__name__})
                target.write(json.dumps(result,ensure_ascii=False)+'\n');target.flush()
                print('Completed',method,'for',row['id'],'status:',result['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--limit',type=int);p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('limit must be positive')
    run(a.split,a.limit,a.out)
