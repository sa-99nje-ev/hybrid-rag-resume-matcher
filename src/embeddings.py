"""Lazy compact encoder with deterministic disk cache and long-input windowing."""
import hashlib
from pathlib import Path
from dotenv import load_dotenv
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / '.env', override=False)
load_dotenv(_PROJECT_ROOT.parent / '.env', override=False)
import numpy as np

class Encoder:
    def __init__(self, model='BAAI/bge-small-en-v1.5', batch_size=16, cache_dir='data/cache'):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_id=model; self.batch_size=batch_size; self.cache=Path(cache_dir)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.model=SentenceTransformer(model,device='cuda' if torch.cuda.is_available() else 'cpu')

    def encode(self, texts, query=False):
        texts=list(texts)
        prefix='Represent this sentence for searching relevant passages: ' if query else ''
        keys=[hashlib.sha256((self.model_id+'\0'+prefix+text).encode('utf-8')).hexdigest() for text in texts]
        unique={}
        for key,text in zip(keys,texts):
            if not (self.cache/(key+'.npy')).exists(): unique[key]=prefix+text
        items=list(unique.items())
        tokenizer=self.model.tokenizer
        maximum=self.model.max_seq_length-tokenizer.num_special_tokens_to_add(pair=False)
        for start in range(0,len(items),self.batch_size):
            batch=items[start:start+self.batch_size]; windows=[]; owners=[]
            for i,(_,value) in enumerate(batch):
                ids=tokenizer.encode(value,add_special_tokens=False,truncation=False)
                if len(ids)<=maximum:
                    windows.append(value); owners.append(i)
                else:
                    for offset in range(0,len(ids),maximum):
                        windows.append(tokenizer.decode(ids[offset:offset+maximum],skip_special_tokens=True)); owners.append(i)
            encoded=self.model.encode(windows,normalize_embeddings=True,convert_to_numpy=True,batch_size=self.batch_size)
            for i,(key,_) in enumerate(batch):
                mean=np.mean(encoded[[j for j,owner in enumerate(owners) if owner==i]],axis=0)
                norm=np.linalg.norm(mean); vector=mean/norm if norm else mean
                np.save(self.cache/(key+'.npy'),vector.astype('float32'))
        return np.stack([np.load(self.cache/(key+'.npy'),allow_pickle=False) for key in keys]) if keys else np.empty((0,0),dtype='float32')
