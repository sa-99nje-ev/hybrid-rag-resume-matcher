"""Index the official FinRank exact-text pool in a separate Qdrant collection."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import configure
configure()
from src.embeddings import Encoder
from src.vector_store import Store
from qdrant_client import models

if __name__=='__main__':
    file=ROOT/'data/processed/finrank_passages.jsonl'
    passages=[json.loads(line) for line in file.open(encoding='utf-8') if line.strip()]
    signature=hashlib.sha256(file.read_bytes()).hexdigest(); stamp=ROOT/'data/processed/finrank_index_signature.json'
    store=Store(collection='finrank_passages'); exists=store.client.collection_exists(store.collection)
    if exists:
        if not stamp.exists() or json.loads(stamp.read_text(encoding='utf-8'))['corpus_sha256']!=signature:
            raise RuntimeError('Existing collection lacks a matching corpus signature; do not overwrite')
        info=store.client.get_collection(store.collection)
        if info.points_count==len(passages):
            print('Reusing existing collection:',info.points_count,'indexed:',info.indexed_vectors_count,'status:',info.status);sys.exit(0)
    encoder=Encoder(cache_dir=str(ROOT/'data/cache/finrank_embeddings'),batch_size=16)
    started=time.perf_counter()
    for offset in range(0,len(passages),64):
        batch=passages[offset:offset+64]; vectors=encoder.encode([p['passage_text'] for p in batch])
        if offset==0:
            store.ensure(vectors.shape[1])
            store.client.update_collection(store.collection,optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000))
            stamp.write_text(json.dumps({'corpus_sha256':signature,'embedding_model':'BAAI/bge-small-en-v1.5'}),encoding='utf-8')
        points=[{'chunk_id':p['passage_id'],'document_id':p['passage_id'],'kind':'financial','text':p['passage_text']} for p in batch]
        store.upsert(points,vectors)
        if (offset//64)%10==0:print('Indexed',min(offset+len(batch),len(passages)),'/',len(passages),flush=True)
    store.client.update_collection(store.collection,optimizers_config=models.OptimizersConfigDiff(indexing_threshold=100))
    print('Upsert time seconds:',round(time.perf_counter()-started,3),'points:',len(passages))
