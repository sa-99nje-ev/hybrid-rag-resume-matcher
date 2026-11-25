import io
import json
from unittest.mock import patch
from main import corpus

def test_corpus_keeps_unicode_line_separators():
    record={'id':'doc','kind':'job','text':'first\u2028second\u0085third'}
    payload=json.dumps(record,ensure_ascii=False)+'\n'
    with patch('pathlib.Path.open',return_value=io.StringIO(payload)):
        documents, parts=corpus('unused.jsonl')
    assert documents[0]['text']==record['text']
    assert ''.join(part['text'] for part in parts)==record['text']
