"""One public BLS question through five retrieval configurations; no relevance labels."""
import json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from main import corpus
from src.embeddings import Encoder
from src.hybrid import rank
from src.vector_store import Store
from src.reranking import Reranker
from src.retrieval import rrf
from src.llm import LLM

QUESTION='What do software quality assurance analysts and testers do?'

def run():
    docs,chunks=corpus(ROOT/'data/processed/djinni_random_1729.jsonl')
    enc=Encoder(cache_dir=str(ROOT/'data/embedding-cache')); store=Store(collection='djinni_random_1729')
    stages={}; timings={}
    for name,mode in [('E1','bm25'),('E2','dense'),('E3','hybrid')]:
        start=time.perf_counter(); stages[name]=rank(QUESTION,docs,enc,store,mode=mode,depth=30,kind='resume'); timings[name]=time.perf_counter()-start
    reranker=Reranker(); start=time.perf_counter(); stages['E4']=reranker.rank(QUESTION,stages['E3'],chunks,depth=10); timings['E4_rerank_only']=time.perf_counter()-start
    start=time.perf_counter(); expansion,usage=LLM().expand(QUESTION)
    rankings=[[d for d,_ in stages['E3']]]
    for q in expansion.queries[:3]: rankings.append([d for d,_ in rank(q,docs,enc,store,mode='hybrid',depth=30,kind='resume')])
    fused=rrf(rankings)[:30]; stages['E5']=reranker.rank(QUESTION,fused,chunks,depth=10); timings['E5_expansion_retrieval_rerank']=time.perf_counter()-start
    details={'query_source':'BLS public-domain occupation question used in prior demonstration','query':QUESTION,'query_count':1,'modes':{k:{'returned':len(v),'top10_document_ids':[x[0] for x in v[:10]]} for k,v in stages.items()},'times_seconds':timings,'query_expansion_usage':usage,'semantic_relevance':'pending independent human judgments','note':'Demonstration only; E4 timing excludes retrieval, E5 timing includes expansion and retrieval; no quality comparison.'}
    target=ROOT/'data/evaluation/ablation_demo_local.json'; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(details,indent=2),encoding='utf-8')
    public={'query_count':1,'returned':{k:len(v) for k,v in stages.items()},'times_seconds':timings,'semantic_relevance':'pending independent human judgments','local_ranking_artifact':str(target.relative_to(ROOT)),'note':details['note']}
    (ROOT/'results/ablation_demo.json').write_text(json.dumps(public,indent=2),encoding='utf-8');print(json.dumps(public,indent=2))
if __name__=='__main__': run()
