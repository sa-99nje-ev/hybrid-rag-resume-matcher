"""Separate FinRank dev RAG pilot. Test generation is intentionally disabled."""
import argparse, collections, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT.parent/'.env',override=False)
os.environ.setdefault('HF_HOME',str(ROOT.parent/'advanced-resume-rag/data/model-cache'))
import numpy as np
from rank_bm25 import BM25Okapi
from src.finrank import load_finrank
from src.embeddings import Encoder
from src.vector_store import Store
from src.retrieval import tokenize,rrf
from src.reranking import Reranker
from src.financial_milestone2 import Provider,citation_status,gate
from src.verification.numerical_verifier import verify,quotation_valid

METHODS=('G1_BM25','G2_Hybrid','G3_Hybrid_CE')
def append(path,item):
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def completed_keys(path):
    complete=set()
    if path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            item=json.loads(line)
            if item['status']!='provider_error':complete.add((item['query_id'],item['method']))
    return complete

def run(split,limit,out):
    if split=='test':raise RuntimeError('Test generation requires a separately frozen configuration and explicit authorization; not implemented in this dev pilot.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/raw/finrank';records,docs,aliases,test,audit=load_finrank(data)
    ids=set(json.loads((data/'baselines/splits.json').read_text(encoding='utf8'))['random'][split])
    queries=sorted((r for r in records if r['id'] in ids),key=lambda r:r['id'])[:limit]
    if not queries:raise ValueError('Empty query set')
    lookup={d['passage_id']:d['passage_text'] for d in docs};docids=list(lookup)
    store=Store(collection='finrank_passages');info=store.client.get_collection(store.collection)
    if info.status.value!='green' or info.points_count!=len(docs) or info.indexed_vectors_count!=len(docs):raise RuntimeError('FinRank Qdrant collection unavailable or incomplete')
    sig=json.loads((ROOT/'data/processed/finrank_index_signature.json').read_text(encoding='utf8'))
    if sig['corpus_sha256']!=hashlib.sha256((ROOT/'data/processed/finrank_passages.jsonl').read_bytes()).hexdigest():raise RuntimeError('Corpus signature mismatch')
    code=[Path(__file__),ROOT/'src/financial_milestone2.py',ROOT/'src/verification/numerical_verifier.py',ROOT/'src/retrieval.py',ROOT/'src/reranking.py',ROOT/'src/embeddings.py']
    config={'split':split,'limit':limit,'query_ids':[q['id'] for q in queries],'dataset_sha256':hashlib.sha256((data/'FinRank.jsonl').read_bytes()).hexdigest(),
      'corpus_sha256':sig['corpus_sha256'],'retrieval':'global pooled 5230 passages, top 20 BM25/dense, RRF k=60, CE rerank','methods':list(METHODS),
      'model':os.getenv('GROQ_MODEL','openai/gpt-oss-20b'),'temperature':0,'max_context_passages':5,'max_context_characters':16000,
      'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
      'attribution':'FinRank (Mansouri et al.), CC BY-NC 4.0; noncommercial research', 'references_sent':False}
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8'))['signature']!=signature:raise RuntimeError('Existing run signature differs; select a new output directory')
    else:manifest.write_text(json.dumps({'signature':signature,'configuration':config},indent=2),encoding='utf8')
    retfile=out/'dev_retrieval.jsonl';ret={}
    if retfile.exists():
        for line in retfile.read_text(encoding='utf8').splitlines():
            r=json.loads(line);ret[r['query_id']]=r
    if any(q['id'] not in ret for q in queries):
        bm=BM25Okapi([tokenize(lookup[pid]) for pid in docids]);encoder=Encoder(cache_dir=str(ROOT/'data/cache/finrank_embeddings'),batch_size=16);reranker=Reranker()
        for q in queries:
            if q['id'] in ret:continue
            scores=np.asarray(bm.get_scores(tokenize(q['question'])));lex=[docids[i] for i in np.argsort(-scores,kind='stable')[:20]]
            vec=encoder.encode([q['question']],query=True)[0];dense=[x.payload['document_id'] for x in store.search(vec,limit=20)]
            hybrid=[x for x,_ in rrf([lex,dense])[:20]];chunks=[{'document_id':p,'text':lookup[p]} for p in hybrid]
            advanced=[p for p,_ in reranker.rank(q['question'],[(x,0.0) for x in hybrid],chunks,depth=20)]
            row={'query_id':q['id'],'regime':'global_pooled_5230','G1_BM25':lex,'G2_Hybrid':hybrid,'G3_Hybrid_CE':advanced};append(retfile,row);ret[q['id']]=row
            print('Prepared dev retrieval',len(ret),'/',len(queries),flush=True)
    pred=out/'dev_predictions.jsonl';rawfile=out/'dev_raw_diagnostics.jsonl';done=set()
    done=completed_keys(pred)
    provider=Provider(config['model'],os.environ['GROQ_API_KEY'])
    for q in queries:
        gold={aliases[p['passage_id']] for p in q['passages']}
        for method in METHODS:
            key=(q['id'],method)
            if key in done:continue
            context=[];chars=0
            for pid in ret[q['id']][method]:
                text=lookup[pid]
                if len(context)>=5:break
                if chars+len(text)>16000:continue
                context.append({'passage_id':pid,'text':text});chars+=len(text)
            evidence={x['passage_id']:x['text'] for x in context}
            row={'query_id':q['id'],'method':method,'regime':'global_pooled_5230','context_ids':list(evidence)}
            if not evidence:row.update(status='no_context',gate={'abstain':True,'reasons':['missing_evidence']})
            else:
                obj,raw,call=provider.request(q['question'],context);raws=[{'request_id':call['request_id'],'raw_response':raw}];calls=[call]
                if obj is None:row.update(status=call['status'],calls=calls)
                else:
                    first=citation_status(obj,evidence);initial=first.copy();repaired=False
                    if not first['quotations_valid'] and not obj.abstention:
                        new,raw2,call2=provider.request(q['question'],context,repair=True);calls.append(call2);raws.append({'request_id':call2['request_id'],'raw_response':raw2})
                        if new is not None:obj=new;repaired=citation_status(obj,evidence)['quotations_valid']
                    final=citation_status(obj,evidence);numerical=verify([x.model_dump() for x in obj.extracted_facts],obj.formula,evidence,question=q['question']);decision=gate(obj,final,numerical)
                    valid_ids={c.passage_id for c in obj.citations if quotation_valid(c.passage_id,c.quotation,evidence)}
                    row.update(status='ok',calls=calls,first_citations=initial,final_citations=final,repaired=repaired,numerical=numerical,gate=decision,
                       model_abstention=obj.abstention,numerical_candidate=bool(obj.extracted_facts or obj.formula),
                       citation_known_gold_precision=(len(valid_ids&gold)/len(valid_ids) if valid_ids else None),
                       citation_known_gold_recall=(len(valid_ids&gold)/len(gold) if gold else None),
                       verified_answer=None if decision['abstain'] else obj.answer)
                for item in raws:append(rawfile,dict(query_id=q['id'],method=method,**item))
            append(pred,row);print('Recorded',method,'status',row['status'],flush=True)
    print('Development run complete.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--limit',type=int,required=True);p.add_argument('--out',default='results/milestone2_dev')
    a=p.parse_args()
    if a.limit<1:p.error('limit must be positive')
    run(a.split,a.limit,a.out)
