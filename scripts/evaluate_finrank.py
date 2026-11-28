"""Evaluate FinRank pooled retrieval and separately its fully judged in-record candidates."""
import csv,hashlib,json,os,sys,time,uuid
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,ndcg_score
from rank_bm25 import BM25Okapi
from qdrant_client import models

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
load_dotenv(ROOT.parent / '.env', override=False)
os.environ.setdefault('HF_HOME',str(ROOT.parent/'advanced-resume-rag/data/model-cache'))
from src.finrank import load_finrank
from src.embeddings import Encoder
from src.vector_store import Store
from src.retrieval import tokenize,rrf
from src.reranking import Reranker

DEPTH=20

def metrics(ranking,gold):
    labels=np.array([1 if x in gold else 0 for x in ranking],dtype=int)
    if not labels.any():raise ValueError('Ranking omits all known gold evidence')
    scores=-np.arange(len(labels),dtype=float)
    out={}
    for k in (5,10,20):
        hits=int(labels[:k].sum());out[f'P@{k}']=hits/k;out[f'R@{k}']=hits/len(gold)
    p=out['P@10'];r=out['R@10'];out['F1@10']=2*p*r/(p+r) if p+r else 0.0
    out['MRR']=1/(int(np.flatnonzero(labels)[0])+1)
    out['MAP']=float(average_precision_score(labels,scores))
    for k in (5,10):out[f'nDCG@{k}']=float(ndcg_score(labels.reshape(1,-1),scores.reshape(1,-1),k=k))
    return out

def cid(p):return str(uuid.uuid5(uuid.NAMESPACE_URL,p))

def run():
    started=time.perf_counter();records,docs,aliases,test,audit=load_finrank(ROOT/'data/raw/finrank')
    docids=[p['passage_id'] for p in docs]; lookup={p['passage_id']:p['passage_text'] for p in docs}; indices={v:i for i,v in enumerate(docids)}
    bm=BM25Okapi([tokenize(p['passage_text']) for p in docs]); enc=Encoder(cache_dir=str(ROOT/'data/cache/finrank_embeddings'),batch_size=16)
    qvecs=enc.encode([r['question'] for r in test],query=True);store=Store(collection='finrank_passages');info=store.client.get_collection(store.collection)
    if info.points_count!=len(docs) or info.status.value!='green' or info.indexed_vectors_count!=len(docs):raise RuntimeError(f'Collection not ready: {info}')
    reranker=Reranker(); output=ROOT/'results';output.mkdir(exist_ok=True)
    rows=[];predictions=[]; counts=defaultdict(list)
    checkpoint=output/'finrank_checkpoint.json'
    signature=hashlib.sha256((Path(__file__).read_bytes()+(ROOT/'src/reranking.py').read_bytes()+(ROOT/'data/raw/finrank/FinRank.jsonl').read_bytes())).hexdigest()
    completed=0
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['signature']!=signature: raise RuntimeError('Checkpoint does not match data and code; preserve it and move it aside before rerunning')
        rows=state['rows']; predictions=state['predictions']; completed=state['completed']
        if len(rows)!=completed*4 or len(predictions)!=completed*4 or any(rows[j*4]['query_id']!=test[j]['id'] for j in range(completed)):
            raise RuntimeError('Checkpoint query order or count mismatch')
        for row in rows:counts[row['method']].append(row)
        print('Resuming after',completed,'queries',flush=True)
    for i,rec in enumerate(test):
        if i<completed:continue
        query=rec['question']; gold={aliases[p['passage_id']] for p in rec['passages']}; neg={aliases[p['passage_id']] for p in rec['hard_negatives']};
        if gold & neg:raise ValueError('Conflicting per-question evidence labels')
        candidates=gold|neg; bm_scores=np.asarray(bm.get_scores(tokenize(query)),dtype=float)
        bm_all=[docids[j] for j in np.argsort(-bm_scores,kind='stable')[:DEPTH]]
        dense_all=[p.payload['document_id'] for p in store.search(qvecs[i],limit=DEPTH)]
        global_lists={'E1_BM25':bm_all,'E2_Dense':dense_all}
        global_lists['E3_Hybrid']=[x for x,_ in rrf([bm_all,dense_all])[:DEPTH]]
        chunks=[{'document_id':p,'text':lookup[p]} for p in global_lists['E3_Hybrid']]
        global_lists['E4_Hybrid_CE']=[p for p,_ in reranker.rank(query,[(x,0.0) for x in global_lists['E3_Hybrid']],chunks,depth=DEPTH)]
        bm_c=sorted(candidates,key=lambda p:(-bm_scores[indices[p]],p))
        f=models.Filter(must=[models.HasIdCondition(has_id=[cid(p) for p in sorted(candidates)])])
        hits=store.client.query_points(store.collection,query=qvecs[i].tolist(),query_filter=f,
                 search_params=models.SearchParams(exact=True),limit=len(candidates),with_payload=True).points
        dense_c=[p.payload['document_id'] for p in hits]
        if set(dense_c)!=candidates:raise RuntimeError('Qdrant did not return all judged candidates')
        ranked={'E1_BM25':bm_c,'E2_Dense':dense_c}
        ranked['E3_Hybrid']=[x for x,_ in rrf([bm_c,dense_c])]
        chunks=[{'document_id':p,'text':lookup[p]} for p in ranked['E3_Hybrid']]
        ranked['E4_Hybrid_CE']=[p for p,_ in reranker.rank(query,[(x,0.0) for x in ranked['E3_Hybrid']],chunks,depth=len(candidates))]
        for method in ranked:
            row={'query_id':rec['id'],'method':method,'regime':'per_question_gold_vs_curated_negatives','gold_count':len(gold),'candidate_count':len(candidates),**metrics(ranked[method],gold)}
            top=global_lists[method]
            for k in (5,10,20):
                row[f'global_known_gold_recall@{k}']=len(set(top[:k])&gold)/len(gold)
                row[f'global_judgment_coverage@{k}']=len(set(top[:k])&candidates)/min(k,len(top)) if top else 0.0
            rows.append(row);counts[method].append(row)
            predictions.append({'query_id':rec['id'],'method':method,'in_record_ranking':ranked[method],'global_top20':top})
        temp=checkpoint.with_suffix('.tmp')
        temp.write_text(json.dumps({'signature':signature,'completed':i+1,'rows':rows,'predictions':predictions}),encoding='utf-8')
        temp.replace(checkpoint)
        if (i+1)%10==0:print(f'Evaluated {i+1}/{len(test)} queries',flush=True)
    keys=['P@5','P@10','R@5','R@10','R@20','F1@10','MRR','MAP','nDCG@5','nDCG@10','global_known_gold_recall@10','global_judgment_coverage@10']
    aggregate=[{'method':m,'queries':len(v),'regime':'per_question_gold_vs_curated_negatives',**{k:float(np.mean([x[k] for x in v])) for k in keys}} for m,v in counts.items()]
    with (output/'finrank_per_query_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (output/'finrank_retrieval_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    with (output/'finrank_retrieval_predictions.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for item in predictions:f.write(json.dumps(item)+'\n')
    rng=np.random.default_rng(42); comparisons={}
    for left,right in [('E3_Hybrid','E1_BM25'),('E3_Hybrid','E2_Dense'),('E4_Hybrid_CE','E3_Hybrid')]:
        diffs=np.array([a['MRR']-b['MRR'] for a,b in zip(counts[left],counts[right])]); samples=rng.choice(diffs,size=(2000,len(diffs)),replace=True).mean(axis=1)
        comparisons[left+' minus '+right]={'metric':'in-record MRR','mean_delta':float(diffs.mean()),'bootstrap_95_percentile_ci':[float(x) for x in np.percentile(samples,[2.5,97.5])]}
    import torch
    manifest={'finrank_git_revision':'916dc27e77b90e294ea1b3a2cc5b0abcf07392d1','data_sha256':hashlib.sha256((ROOT/'data/raw/finrank/FinRank.jsonl').read_bytes()).hexdigest(),
        'audit':audit,'split':'official random test minus cross-split duplicate question','query_count':len(test),'corpus_size':len(docs),
        'models':{'embedding':'BAAI/bge-small-en-v1.5','reranker':'BAAI/bge-reranker-base'},
        'model_revisions':{m:(Path(os.environ['HF_HOME'])/'hub'/('models--'+m.replace('/','--'))/'refs/main').read_text().strip() for m in ['BAAI/bge-small-en-v1.5','BAAI/bge-reranker-base']},
        'versions':{p:version(p) for p in ['sentence-transformers','torch','qdrant-client','rank-bm25','scikit-learn','numpy']},
        'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','torch_cuda':torch.version.cuda,'qdrant':info.config.model_dump(mode='json'),
        'fixed_configuration':{'top_k_global':DEPTH,'rrf_k':60,'cross_encoder':'maximum score across explicit token windows','per_question_candidate_scope':'gold + question-specific curated negatives; Qdrant exact scoring over this judged set'},
        'metric_definitions':'P@k uses k denominator; binary relevance; unjudged global passages are not treated as negatives. Global known-gold recall and judgment coverage are reported separately.',
        'paired_bootstrap_2000_seed_42':comparisons,'elapsed_seconds':time.perf_counter()-started}
    (output/'finrank_experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cols=['method','P@10','R@10','F1@10','MRR','MAP','nDCG@5','nDCG@10']
    lines=['| Configuration | P@10 | R@10 | F1@10 | MRR | MAP | nDCG@5 | nDCG@10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    lines.extend('| '+' | '.join([a['method']]+[f"{a[k]:.4f}" for k in cols[1:]])+' |' for a in aggregate)
    table='\n'.join(lines)
    print('\nPer-question gold-vs-curated-negative reranking; 118 test queries\n'+table,flush=True)
    report='# FinRank retrieval evaluation\n\n'+table+'\n\nThe table ranks each question’s annotated gold passages against its own curated hard negatives. It is not global retrieval or full SEC filing search. Metrics use binary in-record labels; P@k divides by k even when fewer than k candidates exist. Global retrieval is separately reported as known-gold recall and judged coverage in the CSV; unjudged passages are not labeled irrelevant.\n\nOfficial random test contains 119 records; one duplicated question shared with train was excluded before running the models. No train/dev tuning was performed. See PROGRESS.md and the manifest for provenance, split overlaps, all aliases, model revisions and bootstrap intervals.\n'
    (ROOT/'FINRANK_EVALUATION.md').write_text(report,encoding='utf-8')
if __name__=='__main__':run()
