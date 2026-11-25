"""Cross-encoder reranks candidate chunks without silently truncating queries."""
from sentence_transformers import CrossEncoder
import torch

class Reranker:
    def __init__(self, model='BAAI/bge-reranker-base'):
        self.model=CrossEncoder(model,device='cuda' if torch.cuda.is_available() else 'cpu',max_length=512)

    def rank(self,query,candidates,chunks,depth=10):
        ids={doc_id for doc_id,_ in candidates}
        selected=[c for c in chunks if c['document_id'] in ids]
        if not selected: return []
        tok=self.model.tokenizer
        query_ids=tok.encode(query,add_special_tokens=False,truncation=False)
        pairs=[]; owners=[]
        for c in selected:
            doc_ids=tok.encode(c['text'],add_special_tokens=False,truncation=False)
            capacity=512-len(doc_ids)-tok.num_special_tokens_to_add(pair=True)
            if capacity<=0: raise ValueError('A candidate passage exceeds the cross-encoder limit; rechunk it')
            for start in range(0,max(len(query_ids),1),capacity):
                segment=query if len(query_ids)<=capacity else tok.decode(query_ids[start:start+capacity],skip_special_tokens=True)
                pairs.append((segment,c['text'])); owners.append(c['document_id'])
        values=self.model.predict(pairs,batch_size=8,show_progress_bar=False)
        scores={}
        for doc_id,score in zip(owners,values): scores[doc_id]=max(scores.get(doc_id,float('-inf')),float(score))
        return sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:depth]
