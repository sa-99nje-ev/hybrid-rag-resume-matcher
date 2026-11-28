"""Verify the official release and persist an alias-preserving FinRank corpus."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.finrank import load_finrank

if __name__=='__main__':
    records,passages,aliases,test,audit=load_finrank(ROOT/'data/raw/finrank')
    out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    with (out/'finrank_passages.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for p in passages:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    (out/'finrank_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
