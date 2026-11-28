"""Score completed FinRank RAG runs against the original human-authored references."""
import argparse,csv,json,os,sys,statistics
from importlib.metadata import version
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT.parent/'.env',override=False)
os.environ.setdefault('HF_HOME',str(ROOT.parent/'advanced-resume-rag/data/model-cache'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import nltk
nltk.data.path.insert(0,str(ROOT/'data/nltk'))
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import wordpunct_tokenize
from rouge_score import rouge_scorer
from src.finrank import load_finrank

METHODS=('G1_Naive','G2_Hybrid','G3_Advanced')

def run(directory):
    directory=Path(directory);manifest=json.loads((directory/'finrank_rag_manifest.json').read_text(encoding='utf-8'))
    queries=manifest['configuration']['query_ids'];items=[json.loads(s) for s in (directory/'finrank_rag_predictions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    keyed={(r['query_id'],r['method']):r for r in items}
    if len(keyed)!=len(items):raise ValueError('Duplicate query/method output')
    if any((q,m) not in keyed for q in queries for m in METHODS):raise ValueError('Run is incomplete: do not score until all selected queries have attempted outputs')
    recs,_,_,_,_=load_finrank(ROOT/'data/raw/finrank'); refs={r['id']:r['answer'] for r in recs}
    paired=[q for q in queries if all(keyed[q,m]['status']=='ok' and keyed[q,m]['schema_valid'] for m in METHODS)]
    if not paired:raise RuntimeError('No queries have successful outputs for all methods')
    scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True); smooth=SmoothingFunction().method1
    rows=[];cands=[];references=[]
    for q in paired:
        for method in METHODS:
            item=keyed[q,method];answer=item['response']['answer'];reference=refs[q]
            ref_tokens=wordpunct_tokenize(reference.lower());candidate_tokens=wordpunct_tokenize(answer.lower())
            rouge=scorer.score(reference,answer)
            row={'query_id':q,'method':method,'reasoning_type':item['reasoning_type'],
                'BLEU':float(sentence_bleu([ref_tokens],candidate_tokens,smoothing_function=smooth)),
                'ROUGE-1':rouge['rouge1'].fmeasure,'ROUGE-2':rouge['rouge2'].fmeasure,'ROUGE-L':rouge['rougeL'].fmeasure,
                'METEOR':float(meteor_score([ref_tokens],candidate_tokens)),
                'source_reference_valid':item['stats']['source_reference_valid'],
                'gold_evidence_citation_coverage':item['known_gold_citation_coverage'],
                'gold_evidence_context_coverage':item['known_gold_context_coverage'],
                'retrieval_plus_generation_seconds':item.get('retrieval_plus_api_seconds',item.get('end_to_end_seconds')),
                'prompt_tokens':(item['stats']['usage'] or {}).get('prompt_tokens'),
                'completion_tokens':(item['stats']['usage'] or {}).get('completion_tokens')}
            rows.append(row);cands.append(answer);references.append(reference)
    bert_status='pending';bert_hash=None
    try:
        from bert_score import score
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5',local_files_only=True)
        if any(len(tokenizer.encode(s,add_special_tokens=True,truncation=False))>512 for s in cands+references):
            raise ValueError('At least one answer exceeds the 512-token evaluation-model limit')
        (p,r,f),bert_hash=score(cands,references,model_type='BAAI/bge-small-en-v1.5',num_layers=12,
                              idf=False,device='cuda',batch_size=8,return_hash=True,rescale_with_baseline=False)
        for i,row in enumerate(rows):row.update({'BERTScore_P':float(p[i]),'BERTScore_R':float(r[i]),'BERTScore_F1':float(f[i])})
        bert_status='computed'
    except Exception as exc:
        bert_status='unavailable: '+type(exc).__name__
    keys=['BLEU','ROUGE-1','ROUGE-2','ROUGE-L','METEOR','gold_evidence_citation_coverage','gold_evidence_context_coverage','retrieval_plus_generation_seconds']
    if bert_status=='computed':keys+=['BERTScore_P','BERTScore_R','BERTScore_F1']
    aggregate=[]
    for method in METHODS:
        subset=[r for r in rows if r['method']==method];attempted=[keyed[q,method] for q in queries]
        summary={'method':method,'attempted':len(attempted),'schema_valid':sum(bool(x['schema_valid']) for x in attempted),
                 'paired_queries':len(subset),'source_reference_valid_rate':sum(x['source_reference_valid'] for x in subset)/len(subset)}
        summary.update({key:statistics.mean(x[key] for x in subset) for key in keys})
        for field in ('prompt_tokens','completion_tokens'):
            values=[x[field] for x in subset if x[field] is not None]
            summary[field+'_count']=len(values)
            summary[field+'_mean']=statistics.mean(values) if values else None
            summary[field+'_total']=sum(values) if values else None
        aggregate.append(summary)
    for filename,data in [('finrank_generation_per_query.csv',rows),('finrank_generation_metrics.csv',aggregate)]:
        with (directory/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    meta={'split':manifest['configuration']['split'],'attempted_queries':len(queries),'paired_successful_queries':len(paired),
          'note':'Reference-overlap scores are supplementary, not factual accuracy or groundedness. Groundedness and claim support require independent review. Latency is a shared-retrieval component plus API wall time; old pilot files exclude context and local validation overhead. Model initialization is excluded and it is not independently timed end-to-end latency.',
          'bleu':'NLTK sentence BLEU macro-mean, lowercase wordpunct tokens, smoothing method1',
          'rouge':'rouge-score F1, stemming enabled','meteor':'NLTK METEOR, lowercase wordpunct tokens, WordNet',
          'bertscore_status':bert_status,'bertscore_hash':bert_hash,'bertscore_model':'BAAI/bge-small-en-v1.5 (nonstandard BERTScore backbone; existing cache), 12 layers, no IDF or rescaling',
          'versions':{p:version(p) for p in ['nltk','rouge-score','bert-score']}}
    (directory/'finrank_generation_metric_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'metadata':meta,'metrics':aggregate},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();run(a.run_dir)
