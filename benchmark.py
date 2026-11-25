"""Exact FAISS vs server HNSW agreement; never a relevance benchmark."""
import argparse
import json
import statistics
import time
import uuid
from pathlib import Path
import numpy as np
import faiss

from main import ROOT, corpus
from src.embeddings import Encoder
from src.vector_store import Store


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--collection',default='resume_chunks')
    parser.add_argument('--out',default='results/ann.json')
    args=parser.parse_args()
    documents, chunks=corpus(args.corpus)
    resumes=[c for c in chunks if c['kind']=='resume']
    queries=[d['text'] for d in documents if d['kind']=='job']
    if not resumes or not queries: raise ValueError('Need resume chunks and job documents')
    enc=Encoder(cache_dir=str(ROOT/'data/embedding-cache'))
    vectors=enc.encode([c['text'] for c in resumes]); qvec=enc.encode(queries,query=True)
    store=Store(collection=args.collection)
    info=store.client.get_collection(args.collection)
    if not info.indexed_vectors_count:
        raise RuntimeError('Qdrant reports no indexed vectors; HNSW benchmark is not valid')
    # Query length is intentionally checked; embedding model must not silently truncate.
    start=time.perf_counter(); index=faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors); build=time.perf_counter()-start
    recalls={10:[],20:[]}; exact_ms=[]; ann_ms=[]
    for v in qvec:
        start=time.perf_counter(); _, indices=index.search(v.reshape(1,-1), min(20,len(resumes))); exact_ms.append((time.perf_counter()-start)*1000)
        start=time.perf_counter(); hits=store.search(v,limit=min(20,len(resumes)),kind='resume'); ann_ms.append((time.perf_counter()-start)*1000)
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,resumes[i]['chunk_id'])) for i in indices[0] if i>=0]
        observed=[str(p.id) for p in hits]
        for k in recalls:
            exact=set(ids[:k]); recalls[k].append(len(exact.intersection(observed[:k]))/len(exact) if exact else 0)
    result={'query_count':len(queries),'resume_chunks':len(resumes),'qdrant_indexed_vectors':info.indexed_vectors_count,
            'hnsw_config':info.config.hnsw_config.model_dump(), 'faiss_build_seconds':build,
            'ann_recall_at_10':float(np.mean(recalls[10])), 'ann_recall_at_20':float(np.mean(recalls[20])),
            'faiss_ms_p50':float(np.percentile(exact_ms,50)),'faiss_ms_p95':float(np.percentile(exact_ms,95)),
            'qdrant_ms_p50':float(np.percentile(ann_ms,50)),'qdrant_ms_p95':float(np.percentile(ann_ms,95)),
            'note':'Agreement with exact nearest neighbours, not resume-job relevance. Local HTTP overhead included in Qdrant latency.'}
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': run()
