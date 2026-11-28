"""Analyze only the saved development global rankings; no model inference or API calls."""
import json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HOME',str(ROOT.parent/'advanced-resume-rag/data/model-cache'))
from transformers import AutoTokenizer
from src.finrank import load_finrank

def main():
    records,docs,aliases,_,_=load_finrank(ROOT/'data/raw/finrank');byid={r['id']:r for r in records};texts={d['passage_id']:d['passage_text'] for d in docs}
    src=ROOT/'results/milestone2_dev/dev_retrieval.jsonl';rows=[json.loads(x) for x in src.read_text(encoding='utf8').splitlines()]
    tok=AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5',local_files_only=True)
    out={}; ks=(5,8,10,20)
    for method in ('G1_BM25','G2_Hybrid','G3_Hybrid_CE'):
        out[method]={}
        for k in ks:
            recalls=[];sizes=[]
            for row in rows:
                r=byid[row['query_id']]; gold={aliases[p['passage_id']] for p in r['passages']};rank=row[method][:k]
                recalls.append(len(gold.intersection(rank))/len(gold))
                payload=json.dumps({'question':r['question'],'evidence':[{'passage_id':pid,'text':texts[pid]} for pid in rank]},ensure_ascii=False)
                sizes.append(len(tok.encode(payload,add_special_tokens=False,truncation=False,verbose=False)))
            out[method][str(k)]={'n':len(rows),'mean_known_gold_recall':sum(recalls)/len(recalls),'mean_proxy_tokens':sum(sizes)/len(sizes)}
    artifact={'retrieval_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scope':'10 saved dev queries; global 5230 passage pool',
      'token_note':'BGE tokenizer counts JSON question and full passages only, excluding system/schema and provider tokenization; proxy estimates, not Groq billed prompt tokens.',
      'warning':'Known-gold recall is incomplete judgment coverage, not semantic relevance.', 'methods':out}
    dest=ROOT/'results/milestone2_reliability_dev';dest.mkdir(parents=True,exist_ok=True)
    (dest/'context_depth_analysis.json').write_text(json.dumps(artifact,indent=2),encoding='utf8');print(json.dumps(artifact,indent=2))
if __name__=='__main__':main()
