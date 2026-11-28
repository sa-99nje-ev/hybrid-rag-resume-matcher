"""Summarize local reliability results without printing source passages or answers."""
import json,collections,statistics
from pathlib import Path

def main():
    root=Path('results/milestone2_reliability_dev/run5');rows=[json.loads(x) for x in (root/'predictions.jsonl').read_text(encoding='utf8').splitlines()];output=[]
    for m in ('G1_BM25','G2_Hybrid','G3_Hybrid_CE'):
        group=[r for r in rows if r['method']==m];ok=[r for r in group if r['status']=='ok'];calls=[r['provider'] for r in group if 'provider' in r]
        output.append({'method':m,'recorded':len(group),'schema_valid':len(ok),'provider_failures':sum(r['status']=='provider_error' for r in group),
          'first_valid':sum(r['first_citations']['quotations_valid'] for r in ok),'after_local_alignment_valid':sum(r['final_citations']['quotations_valid'] for r in ok),
          'aligned_citations':sum(x['collection']=='citations' and x['status']=='aligned_punctuation_only' for r in ok for x in r['citation_audit']),
          'numerical_verified_answer':sum(r['numerical']['status']=='VERIFIED' and r['numerical']['answer_status']=='MATCH' for r in ok),
          'abstentions':sum(r['gate']['abstain'] for r in ok),'prompt_tokens':sum((c.get('usage') or {}).get('prompt_tokens',0) or 0 for c in calls),
          'completion_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) or 0 for c in calls),
          'api_latency_median':statistics.median([c['latency_seconds'] for c in calls]) if calls else None,
          'est_new':sum(r['new_prompt_estimate'] for r in group),'est_prior':sum(r['prior_context_prompt_estimate'] for r in group)})
    reasons=collections.Counter(reason for r in rows if r['status']=='ok' for reason in r['gate']['reasons'])
    events=collections.Counter(e for r in rows for e in r.get('provider',{}).get('events',[]))
    summary={'methods':output,'abstention_reasons':dict(reasons),'provider_events':dict(events),'recorded':len(rows),
      'limitations':'Source span and Decimal checks do not establish semantic or financial correctness. Token estimates use UTF-8 bytes/3, not the Groq tokenizer. Prompt estimates use the same new instruction/schema to isolate context effects.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
