"""Export real candidate pairs for human judgment; no relevance labels are generated."""
import csv
from pathlib import Path
from main import corpus, ROOT
from src.embeddings import Encoder
from src.vector_store import Store
from src.hybrid import rank

def run():
    documents,_=corpus(ROOT/'data/processed/documents.jsonl')
    lookup={d['id']:d for d in documents}
    encoder=Encoder(cache_dir=str(ROOT/'data/embedding-cache')); store=Store()
    path=ROOT/'data/evaluation/annotation_candidates.csv'; path.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['job_id','resume_id','job_text','resume_text','human_grade','annotator','notes'])
        writer.writeheader()
        for job in (d for d in documents if d['kind']=='job'):
            for doc_id,_ in rank(job['text'],documents,encoder,store,kind='resume',depth=10):
                writer.writerow({'job_id':job['id'],'resume_id':doc_id,'job_text':job['text'],'resume_text':lookup[doc_id]['text']})
                count+=1
    print('Unlabeled candidate pairs:',count,'saved locally to',path)

if __name__=='__main__': run()
