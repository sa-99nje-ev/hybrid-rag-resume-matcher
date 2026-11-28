"""Summarize completed milestone 2 attempts without reading or exposing model text."""
import argparse,collections,csv,json,statistics
from pathlib import Path

def quantile(values,q):
    if not values:return None
    values=sorted(values);i=(len(values)-1)*q;lo=int(i);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(i-lo)

def summarize(out):
    out=Path(out);pred=out/'dev_predictions.jsonl'; latest={}
    for line in pred.read_text(encoding='utf8').splitlines():
        r=json.loads(line);latest[(r['query_id'],r['method'])]=r
    rows=list(latest.values());summaries=[]
    for method in ('G1_BM25','G2_Hybrid','G3_Hybrid_CE'):
        group=[r for r in rows if r['method']==method];ok=[r for r in group if r['status']=='ok'];calls=[c for r in group for c in r.get('calls',[])]
        latency=[sum(c['latency_seconds'] for c in r.get('calls',[])) for r in group if r.get('calls')]
        def mean(vals):return statistics.mean(vals) if vals else None
        summaries.append({'method':method,'recorded':len(group),'schema_valid':len(ok),'provider_errors':sum(r['status']=='provider_error' for r in group),
            'schema_invalid':sum(r['status']=='schema_invalid' for r in group),'first_quote_valid':sum(r['first_citations']['quotations_valid'] for r in ok),
            'final_quote_valid':sum(r['final_citations']['quotations_valid'] for r in ok),'final_passage_id_valid':sum(r['final_citations']['passage_ids_valid'] for r in ok),
            'repaired':sum(r['repaired'] for r in ok),'numerical_candidate':sum(r['numerical_candidate'] for r in ok),
            'numerical_verified':sum(r['numerical']['status']=='VERIFIED' for r in ok),'abstained':sum(r['gate']['abstain'] for r in ok),
            'known_gold_citation_precision_mean':mean([r['citation_known_gold_precision'] for r in ok if r['citation_known_gold_precision'] is not None]),
            'known_gold_citation_recall_mean':mean([r['citation_known_gold_recall'] for r in ok if r['citation_known_gold_recall'] is not None]),
            'prompt_tokens':sum((c.get('usage') or {}).get('prompt_tokens',0) or 0 for c in calls),
            'completion_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) or 0 for c in calls),
            'call_latency_p50_seconds':quantile(latency,.5),'call_latency_p95_seconds':quantile(latency,.95)})
    with (out/'dev_diagnostics.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0].keys());w.writeheader();w.writerows(summaries)
    counts={'rows':len(rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),
      'verification_statuses':dict(collections.Counter(r['numerical']['status'] for r in rows if r['status']=='ok')),
      'gate_reasons':dict(collections.Counter(reason for r in rows if r['status']=='ok' for reason in r['gate']['reasons'])),
      'caveat':'Known-gold ID overlap is not semantic support; unjudged citations are not independently established as irrelevant. Provider failures are not abstentions. Latency is generation API wall time, excludes retrieval/model initialization.'}
    (out/'dev_diagnostics.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
    print(json.dumps({'counts':counts,'methods':summaries},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/milestone2_dev');summarize(p.parse_args().out)
