"""Compare CPU/GPU model inference on the same authentic local passages."""
import gc
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from main import ROOT, corpus
from sentence_transformers import SentenceTransformer, CrossEncoder

def timing(fn, device, repeats=3):
    fn()
    if device=='cuda': torch.cuda.synchronize()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter(); fn()
        if device=='cuda': torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return {'runs_ms':values,'median_ms':statistics.median(values)}

def run():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    docs,parts=corpus(ROOT/'data/processed/djinni_benchmark.jsonl')
    passages=[c['text'] for c in parts if c['kind']=='resume'][:8]
    query=next(d['text'] for d in docs if d['kind']=='job')[:250]
    result={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0),
            'samples':len(passages),'repeats':3,'note':'One warmup and three measured calls; includes tokenization and model inference; excludes model loading.'}
    for mode in ['embedding','reranking']:
        result[mode]={}
        for device in ['cpu','cuda']:
            if device=='cuda': torch.cuda.reset_peak_memory_stats()
            model=SentenceTransformer('BAAI/bge-small-en-v1.5',device=device) if mode=='embedding' else CrossEncoder('BAAI/bge-reranker-base',device=device,max_length=512)
            if mode=='embedding': fn=lambda:model.encode(passages,batch_size=8,normalize_embeddings=True,show_progress_bar=False)
            else: fn=lambda:model.predict([(query,text) for text in passages],batch_size=8,show_progress_bar=False)
            result[mode][device]=timing(fn,device)
            if device=='cuda': result[mode][device]['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            del fn,model; gc.collect()
            if device=='cuda': torch.cuda.empty_cache()
    path=ROOT/'results/device_comparison.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
