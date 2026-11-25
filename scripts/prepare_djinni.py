"""Stream the authors' English Djinni Parquet files into the project JSONL format."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]

def convert(destination, resume_limit=None, job_limit=None):
    specifications=[('resume','candidates','CV',resume_limit),('job','jobs','Long Description',job_limit)]
    seen=set(); counts={}; skipped={}
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w',encoding='utf-8',newline='\n') as out:
        for kind,folder,column,limit in specifications:
            path=ROOT/'data/raw/djinni'/folder/'data/train-00000-of-00001.parquet'
            parquet=pq.ParquetFile(path); total=0; invalid=0
            for batch in parquet.iter_batches(batch_size=512,columns=['id',column,'Position','Primary Keyword']):
                for row in batch.to_pylist():
                    if limit is not None and total>=limit: break
                    text=row[column]; source_id=row['id']
                    if not isinstance(text,str) or not text.strip() or not isinstance(source_id,str) or not source_id:
                        invalid+=1; continue
                    doc_id='djinni:'+kind+':'+source_id
                    if doc_id in seen: raise ValueError('Repeated upstream ID: '+doc_id)
                    seen.add(doc_id)
                    record={'id':doc_id,'kind':kind,'text':text,'source':'lang-uk Djinni '+folder,
                            'source_id':source_id,'position':row['Position'],'primary_keyword':row['Primary Keyword']}
                    out.write(json.dumps(record,ensure_ascii=False)+'\n'); total+=1
                if limit is not None and total>=limit: break
            counts[kind]=total; skipped[kind]=invalid
    return {'written':counts,'skipped_invalid':skipped,'selection':'first valid records in upstream Parquet order',
            'resume_limit':resume_limit,'job_limit':job_limit}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--resume-limit',type=int); p.add_argument('--job-limit',type=int)
    a=p.parse_args()
    if any(n is not None and n<=0 for n in [a.resume_limit,a.job_limit]): p.error('limits must be positive')
    print(json.dumps(convert(a.output,a.resume_limit,a.job_limit)))
