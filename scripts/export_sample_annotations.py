"""Export real job/candidate pairs for independent human labeling, never fill judgments."""
import csv, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from main import corpus
from src.embeddings import Encoder
from src.hybrid import rank
from src.vector_store import Store

def run():
    docs,_=corpus(ROOT/'data/processed/djinni_random_1729.jsonl'); lookup={d['id']:d for d in docs}
    jobs=random.Random(1729).sample([d for d in docs if d['kind']=='job'],10)
    enc=Encoder(cache_dir=str(ROOT/'data/embedding-cache')); store=Store(collection='djinni_random_1729')
    target=ROOT/'data/evaluation/djinni_random_annotation.csv';target.parent.mkdir(parents=True,exist_ok=True); count=0
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        fields=['job_id','resume_id','job_text','resume_text','relevance_grade','annotator','notes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for job in jobs:
            for did,_ in rank(job['text'],docs,enc,store,mode='hybrid',depth=5,kind='resume'):
                w.writerow({'job_id':job['id'],'resume_id':did,'job_text':job['text'],'resume_text':lookup[did]['text'],'relevance_grade':'','annotator':'','notes':''});count+=1
    print('Local candidate pairs:',count,'blank judgment fields; file:',target)
if __name__=='__main__': run()
