"""Exact and high-similarity duplicate screening; no relevance judgments."""
import argparse, hashlib, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

def normalize(text): return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def tokens(text): return re.findall(r'(?u)\w+|[+#./-]+',normalize(text))
def shingles(text):
    words=tokens(text)
    return {' '.join(words[i:i+3]) for i in range(max(1,len(words)-2))}
def simhash(values):
    weights=[0]*64
    for s in values:
        h=int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'big')
        for bit in range(64): weights[bit]+=1 if h & (1<<bit) else -1
    return sum((1<<i) for i,v in enumerate(weights) if v>=0)

def audit(records,old=None):
    exact=defaultdict(list); buckets=defaultdict(list); signatures={}; sets={}; candidates=set(); by_id={r['id']:r for r in records}
    for r in records:
        did=r['id']; kind=r['kind']; text=normalize(r['text']); digest=hashlib.sha256(text.encode()).hexdigest(); exact[(kind,digest)].append(did)
        sh=shingles(text); sig=simhash(sh); signatures[did]=sig; sets[did]=sh
        for band in range(4):
            key=(kind,band,(sig>>(16*band))&65535)
            for other in buckets[key]: candidates.add(tuple(sorted((did,other))))
            buckets[key].append(did)
    pairs=[]
    for a,b in sorted(candidates):
        if (signatures[a]^signatures[b]).bit_count()>3: continue
        union=sets[a]|sets[b]; score=len(sets[a]&sets[b])/len(union) if union else 1.0
        if score>=0.8 and normalize(by_id[a]['text'])!=normalize(by_id[b]['text']): pairs.append((a,b,round(score,4)))
    groups=[v for v in exact.values() if len(v)>1]
    by_text=defaultdict(set)
    for record in records: by_text[hashlib.sha256(normalize(record['text']).encode()).hexdigest()].add(record['kind'])
    cross_kind_exact=sum(len(kinds)>1 for kinds in by_text.values())
    result={'cross_kind_exact_text_groups':cross_kind_exact,'documents':len(records),'kinds':dict(Counter(r['kind'] for r in records)), 'exact_groups':len(groups),'exact_duplicate_documents':sum(len(x)-1 for x in groups),'near_duplicate_pairs_found':len(pairs), 'candidate_pairs':len(candidates), 'method':'NFKC/casefold/whitespace exact; 3-token shingle Jaccard >= 0.8 after 64-bit SimHash candidate filter with Hamming distance <= 3','limitation':'Near-duplicate candidates outside the SimHash distance threshold are missed; no guarantee of exhaustiveness. No relevance labels inferred.'}
    if old is not None:
        oldids={r['id'] for r in old}; result['source_order_previous_id_overlap']=sum(r['id'] in oldids for r in records)
    return result,pairs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--previous');p.add_argument('--output',required=True);a=p.parse_args()
    def read(path):
        with open(path,encoding='utf-8-sig') as f:return [json.loads(line) for line in f if line.strip()]
    summary,pairs=audit(read(a.corpus),read(a.previous) if a.previous else None)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    local=Path('data/evaluation/duplicate_pairs.json');local.parent.mkdir(parents=True,exist_ok=True);local.write_text(json.dumps(pairs),encoding='utf-8')
    print(json.dumps(summary,indent=2))
