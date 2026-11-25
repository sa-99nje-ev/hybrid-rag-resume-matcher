"""Uniform without-replacement sample of original Djinni Parquet row positions."""
import argparse, hashlib, json, random, sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]

def sample(out, seed, resume_count, job_count):
    specs=[('resume','candidates','CV',resume_count),('job','jobs','Long Description',job_count)]
    manifest={'seed':seed,'algorithm':'Python random.Random.sample of upstream row positions; sorted output','python':sys.version.split()[0],'sources':{},'counts':{}}
    target=Path(out); target.parent.mkdir(parents=True,exist_ok=True); seen=set()
    with target.open('w',encoding='utf-8',newline='\n') as dest:
        for kind,folder,column,count in specs:
            path=ROOT/'data/raw/djinni'/folder/'data/train-00000-of-00001.parquet'
            parquet=pq.ParquetFile(path); total=parquet.metadata.num_rows
            positions=set(random.Random(f'{seed}:{kind}').sample(range(total),count)); selected=[]; offset=0
            manifest['sources'][kind]={'total_rows':total,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
            for batch in parquet.iter_batches(batch_size=1024,columns=['id',column,'Position','Primary Keyword']):
                for i,row in enumerate(batch.to_pylist()):
                    pos=offset+i
                    if pos not in positions: continue
                    source_id=row['id']; text=row[column]
                    if not isinstance(source_id,str) or not source_id or not isinstance(text,str) or not text.strip():
                        raise ValueError(f'Selected invalid {kind} record at row {pos}; no substitution made')
                    did=f'djinni:{kind}:{source_id}'
                    if did in seen: raise ValueError('Duplicate source ID in sampled documents')
                    seen.add(did); selected.append(pos)
                    dest.write(json.dumps({'id':did,'kind':kind,'text':text,'source':'lang-uk Djinni '+folder,'source_id':source_id,'source_row':pos,'position':row['Position'],'primary_keyword':row['Primary Keyword']},ensure_ascii=False)+'\n')
                offset+=batch.num_rows
            if len(selected)!=count: raise ValueError('Sample count mismatch')
            manifest['counts'][kind]=len(selected)
            manifest['sources'][kind]['selected_positions_sha256']=hashlib.sha256(','.join(map(str,selected)).encode()).hexdigest()
    m=target.with_suffix('.manifest.json'); m.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=1729); p.add_argument('--resumes',type=int,default=10000); p.add_argument('--jobs',type=int,default=200); a=p.parse_args(); sample(a.output,a.seed,a.resumes,a.jobs)
