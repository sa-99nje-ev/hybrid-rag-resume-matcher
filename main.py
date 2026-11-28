"""CLI for locally authorized financial documents and evidence retrieval."""
import argparse,json,os
from pathlib import Path
from src.config import configure
from src.preprocessing import chunks
ROOT=Path(__file__).resolve().parent
configure()

def corpus(path):
    documents=[];parts=[];seen=set()
    with Path(path).open(encoding='utf-8-sig') as file:
        for line in file:
            if not line.strip():continue
            d=json.loads(line);did=d.get('id');text=d.get('text')
            if not isinstance(did,str) or not did or did in seen or d.get('kind')!='financial' or not isinstance(text,str) or not text:
                raise ValueError('Each record needs a unique ID, financial kind and nonempty text')
            seen.add(did);documents.append(d)
            for item in chunks(did,text):item['kind']='financial';parts.append(item)
    return documents,parts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--corpus',required=True);parser.add_argument('--collection',default='financial_passages')
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('index')
    search=sub.add_parser('search');search.add_argument('question');search.add_argument('--mode',choices=['bm25','dense','hybrid'],default='hybrid')
    ask=sub.add_parser('ask');ask.add_argument('question');ask.add_argument('--allow-external-data',action='store_true');ask.add_argument('--diagnostics')
    args=parser.parse_args();docs,parts=corpus(args.corpus)
    from src.embeddings import Encoder
    from src.vector_store import Store
    encoder=Encoder();store=Store(collection=args.collection)
    if args.command=='index':
        vectors=encoder.encode([p['text'] for p in parts]);store.ensure(vectors.shape[1]);store.upsert(parts,vectors);print('Indexed passages:',len(parts));return
    if args.command=='search':
        from src.hybrid import rank
        print(json.dumps(rank(args.question,docs,encoder,store,mode=args.mode,kind='financial'),indent=2));return
    if not args.allow_external_data:parser.error('External data transmission requires --allow-external-data after reviewing source rights')
    from src.reranking import Reranker
    from src.reliability_provider import ReliabilityProvider
    from src.financial_pipeline import answer
    provider=ReliabilityProvider(os.getenv('GROQ_MODEL','openai/gpt-oss-20b'),os.environ['GROQ_API_KEY'])
    result=answer(args.question,docs,parts,encoder,store,Reranker(),provider,diagnostics_path=args.diagnostics)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
