"""CLI for a local corpus of JSONL documents: id, kind (resume/job), text."""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
load_dotenv(ROOT.parent / '.env')
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'model-cache'))

from src.preprocessing import chunks


def corpus(path):
    with Path(path).open(encoding='utf-8-sig') as source:
        documents = [json.loads(line) for line in source if line.strip()]
    ids = set()
    for d in documents:
        if not isinstance(d.get('id'), str) or not d['id'] or d['id'] in ids or d.get('kind') not in {'resume', 'job'} or not isinstance(d.get('text'), str) or not d['text']:
            raise ValueError('Each document needs a unique nonempty string id, kind resume/job, and text')
        ids.add(d['id'])
    parts = []
    for d in documents:
        parts.extend(dict(c, kind=d['kind']) for c in chunks(d['id'], d['text'], size=700, overlap=100))
    return documents, parts


def run():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus', required=True, help='Document JSONL path')
    p.add_argument('--collection', default='resume_chunks')
    sub=p.add_subparsers(dest='command', required=True)
    sub.add_parser('index')
    search=sub.add_parser('search'); search.add_argument('query'); search.add_argument('--kind',choices=['resume','job']); search.add_argument('--mode',choices=['bm25','dense','hybrid'],default='hybrid'); search.add_argument('--rerank',action='store_true')
    ask=sub.add_parser('ask'); ask.add_argument('query'); ask.add_argument('--kind',choices=['resume','job']); ask.add_argument('--mode',choices=['naive','hybrid','advanced'],default='advanced'); ask.add_argument('--allow-external-data',action='store_true')
    ask.add_argument('--prompt-style',choices=['zero','few','stepwise'],default='zero'); ask.add_argument('--examples')
    agent=sub.add_parser('agent'); agent.add_argument('query'); agent.add_argument('--allow-external-data',action='store_true')
    args=p.parse_args(); docs, parts=corpus(args.corpus)
    from src.embeddings import Encoder
    from src.vector_store import Store
    from src.hybrid import rank
    encoder=Encoder(cache_dir=str(ROOT / 'data' / 'embedding-cache'))
    store=Store(os.getenv('QDRANT_URL','http://localhost:6333'),args.collection)
    if args.command == 'index':
        if not parts: raise ValueError('Empty corpus')
        for i in range(0,len(parts),32):
            batch=parts[i:i+32]; vectors=encoder.encode([c['text'] for c in batch])
            if i == 0: store.ensure(vectors.shape[1])
            store.upsert(batch,vectors)
        print(json.dumps({'documents':len(docs),'chunks':len(parts),'collection':args.collection})); return
    if args.command == 'agent':
        if not args.allow_external_data:
            raise ValueError('Check document privacy and license, then pass --allow-external-data')
        from src.llm import LLM
        from src.agent import run_agent
        from src.generation import answer
        from src.rag import verify_citation
        llm=LLM(); lookup={d['id']:d['text'] for d in docs}; by_chunk={c['chunk_id']:c for c in parts}
        def search_kind(query, kind):
            return [{'document_id':doc_id,'score':score} for doc_id,score in rank(query,docs,encoder,store,kind=kind,depth=10)]
        def get_sections(document_id):
            return [c for c in parts if c['document_id']==document_id]
        def compare_resume_job(resume_id, job_id):
            if resume_id not in lookup or job_id not in lookup: raise ValueError('Unknown document')
            return answer(lookup[job_id],[(resume_id,1)],docs,parts,llm)
        functions={'search_resumes':lambda query:search_kind(query,'resume'),
                   'search_jobs':lambda query:search_kind(query,'job'), 'get_sections':get_sections,
                   'compare_resume_job':compare_resume_job,
                   'verify_citation':lambda document_id,chunk_id,quote:verify_citation({'document_id':document_id,'chunk_id':chunk_id,'quote':quote},lookup,by_chunk)}
        print(json.dumps(run_agent(llm,args.query,functions,verify_final=functions['verify_citation']),ensure_ascii=True,indent=2)); return
    mode = args.mode if args.command == 'search' else ('dense' if args.mode == 'naive' else 'hybrid')
    if args.command == 'ask' and not args.allow_external_data:
        raise ValueError('Check document privacy and license, then pass --allow-external-data to send evidence to Groq')
    ranking=rank(args.query,docs,encoder,store,mode=mode,kind=args.kind)
    llm=None
    query_analysis=None
    expansion_stats=None
    if args.command == 'ask' and args.mode == 'advanced':
        from src.llm import LLM
        from src.retrieval import rrf
        llm=LLM(); expanded,expansion_stats=llm.expand(args.query); query_analysis=expanded.model_dump()
        rankings=[[d for d,_ in ranking]]
        for query in expanded.queries[:3]: rankings.append([d for d,_ in rank(query,docs,encoder,store,mode='hybrid',kind=args.kind)])
        ranking=rrf(rankings)[:30]
    if (args.command == 'search' and args.rerank) or (args.command == 'ask' and args.mode == 'advanced'):
        from src.reranking import Reranker
        ranking=Reranker().rank(args.query,ranking,parts)
    if args.command == 'search': print(json.dumps(ranking,ensure_ascii=True)); return
    from src.llm import LLM
    from src.generation import answer
    examples=json.loads(Path(args.examples).read_text(encoding='utf-8')) if args.examples else None
    result=answer(args.query,ranking,docs,parts,llm or LLM(),prompt_style=args.prompt_style,examples=examples)
    result['query_analysis']=query_analysis
    result['expansion_stats']=expansion_stats
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__ == '__main__': run()
