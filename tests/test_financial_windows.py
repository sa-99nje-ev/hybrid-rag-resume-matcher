from src.reranking import Reranker

class ExpandingTokenizer:
    def encode(self,text,text_pair=None,**kwargs):
        if text_pair is not None:return list(range(513 if len(text_pair)>200 else 100))
        return list(range(3 if text=='Q' else 500))
    def decode(self,ids,**kwargs):return 'x'*len(ids)
    def num_special_tokens_to_add(self,pair=False):return 3 if pair else 2

class FakeCrossEncoder:
    tokenizer=ExpandingTokenizer()
    def predict(self,pairs,**kwargs):
        self.pairs=pairs
        assert all(len(pair[1])<=200 for pair in pairs)
        return [1.0]*len(pairs)

def test_reranker_splits_retokenized_overflow_without_dropping_windows():
    ranker=Reranker.__new__(Reranker);ranker.model=FakeCrossEncoder()
    result=ranker.rank('Q',[('a',1)],[{'document_id':'a','text':'evidence'}])
    assert result==[('a',1.0)]
    assert sum(len(pair[1]) for pair in ranker.model.pairs)==500
