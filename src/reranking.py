"""Cross-encoder reranking with explicit query and passage token windows."""
from sentence_transformers import CrossEncoder
import torch

class Reranker:
    def __init__(self, model='BAAI/bge-reranker-base'):
        self.model=CrossEncoder(model,device='cuda' if torch.cuda.is_available() else 'cpu',max_length=512)

    def rank(self,query,candidates,chunks,depth=10):
        ids={doc_id for doc_id,_ in candidates}
        selected=[c for c in chunks if c['document_id'] in ids]
        if not selected:return []
        tok=self.model.tokenizer; query_ids=tok.encode(query,add_special_tokens=False,truncation=False)
        query_windows=[query_ids[i:i+128] for i in range(0,len(query_ids),128)] or [[]]
        pairs=[]; owners=[]
        for c in selected:
            doc_ids=tok.encode(c['text'],add_special_tokens=False,truncation=False)
            for q_ids in query_windows:
                capacity=512-len(q_ids)-tok.num_special_tokens_to_add(pair=True)
                for start in range(0,max(len(doc_ids),1),capacity):
                    pending=[(q_ids,doc_ids[start:start+capacity])]
                    while pending:
                        q_part,d_part=pending.pop()
                        q=tok.decode(q_part,skip_special_tokens=True); text=tok.decode(d_part,skip_special_tokens=True)
                        if len(tok.encode(q,text,add_special_tokens=True,truncation=False))>512:
                            if len(d_part)>1:
                                mid=len(d_part)//2; pending.extend([(q_part,d_part[mid:]),(q_part,d_part[:mid])])
                            elif len(q_part)>1:
                                mid=len(q_part)//2; pending.extend([(q_part[mid:],d_part),(q_part[:mid],d_part)])
                            else: raise ValueError('Unable to fit a single-token query/passage pair')
                        else:
                            pairs.append((q,text));owners.append(c['document_id'])
        values=self.model.predict(pairs,batch_size=8,show_progress_bar=False)
        scores={}
        for doc_id,score in zip(owners,values):scores[doc_id]=max(scores.get(doc_id,float('-inf')),float(score))
        return sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:depth]
