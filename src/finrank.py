"""FinRank release loading, exact-text pooling, and provenance checks."""
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def load_finrank(directory):
    directory=Path(directory)
    with (directory/'FinRank.jsonl').open(encoding='utf-8') as f:
        records=[json.loads(line) for line in f if line.strip()]
    splits=json.loads((directory/'baselines/splits.json').read_text(encoding='utf-8'))['random']
    ids={r['id'] for r in records}
    if set().union(*(set(v) for v in splits.values())) != ids or sum(map(len,splits.values())) != len(ids):
        raise ValueError('Official random split does not partition records')
    passages=[]; by_text={}; alias_to_canonical={}; hash_errors=[]
    for record in records:
        for key in ('passages','hard_negatives'):
            for passage in record[key]:
                text=passage['text']; pid=passage['passage_id']
                calculated=hashlib.sha1(unicodedata.normalize('NFC',text.strip()).encode('utf-8')).hexdigest()
                if calculated!=passage['text_sha1']: hash_errors.append(pid)
                if text not in by_text:
                    by_text[text]=len(passages)
                    passages.append({'passage_id':pid,'passage_text':text,'text_hash':hashlib.sha256(text.encode('utf-8')).hexdigest(),'aliases':[],'sources':[]})
                item=passages[by_text[text]]; alias_to_canonical[pid]=item['passage_id']; item['aliases'].append(pid)
                item['sources'].append({'original_passage_id':pid,'original_text_sha1':passage['text_sha1'],
                    'record_id':record['id'],'role':'gold' if key=='passages' else 'hard_negative',
                    'ticker':record.get('ticker') if key=='passages' else passage.get('ticker'),
                    'fiscal_year':record.get('year') if key=='passages' else passage.get('year'),
                    'filing_type':record.get('doc_type') if key=='passages' else passage.get('doc_type'),
                    'document_id':record.get('folder_name') if key=='passages' else None,
                    'page_number':passage.get('page_number')})
    if hash_errors: raise ValueError(f'{len(hash_errors)} passage hashes do not match NFC/strip SHA-1')
    if len(alias_to_canonical)!=sum(len(r['passages'])+len(r['hard_negatives']) for r in records):raise ValueError('Repeated source passage ID')
    questions=defaultdict(set); membership={rid:split for split,items in splits.items() for rid in items}
    for r in records:questions[' '.join(r['question'].casefold().split())].add(membership[r['id']])
    overlap={q for q,groups in questions.items() if 'test' in groups and len(groups)>1}
    test=[r for r in records if membership[r['id']]=='test' and ' '.join(r['question'].casefold().split()) not in overlap]
    gold_by_split={name:{alias_to_canonical[p['passage_id']] for r in records if membership[r['id']]==name for p in r['passages']} for name in splits}
    gold_texts={p['text'].strip() for r in records for p in r['passages']}
    audit={'records':len(records),'corpus_unique_exact_texts':len(passages),'source_passage_occurrences':len(alias_to_canonical),
           'gold_occurrences':sum(len(r['passages']) for r in records),'negative_occurrences':sum(len(r['hard_negatives']) for r in records),
           'negative_occurrences_matching_any_gold_stripped':sum(p['text'].strip() in gold_texts for r in records for p in r['hard_negatives']),
           'hash_algorithm':'SHA1(NFC(strip(text)).encode(UTF-8))','hash_errors':len(hash_errors),
           'splits':{k:len(v) for k,v in splits.items()},'test_excluded_duplicate_question':len(splits['test'])-len(test),
           'test_evaluated':len(test),'duplicate_question_groups_crossing_splits':len(overlap),
           'gold_text_overlap_train_test':len(gold_by_split['train'] & gold_by_split['test']),
           'gold_text_overlap_dev_test':len(gold_by_split['dev'] & gold_by_split['test']),
           'tickers':len({r['ticker'] for r in records}),
           'note':'Exact-text pooling follows upstream first-occurrence corpus; all source IDs and provenance are retained as aliases. Cross-query gold/negative reuse does not make a negative positive for a different question.'}
    return records,passages,alias_to_canonical,test,audit
