"""Financial retrieval, reranking and one-call evidence planning."""
from .hybrid import rank
from .financial_pot import answer_with_context,fail

def answer(question,documents,chunks,encoder,store,reranker,provider,depth=20,max_passages=8,max_chars=15000,diagnostics_path=None):
    if not question.strip():return fail('MISSING_QUESTION')
    ranking=rank(question,documents,encoder,store,mode='hybrid',depth=depth,kind='financial')
    reranked=reranker.rank(question,ranking,chunks,depth=depth)
    rank_index={doc_id:i for i,(doc_id,_) in enumerate(reranked)}
    selected=sorted((c for c in chunks if c['document_id'] in rank_index),key=lambda c:(rank_index[c['document_id']],c.get('start',0)))
    context=[];size=0;omitted=[]
    for part in selected:
        if len(context)>=max_passages:break
        if size+len(part['text'])>max_chars:omitted.append(part['chunk_id']);continue
        context.append({'passage_id':part['chunk_id'],'text':part['text']});size+=len(part['text'])
    result=answer_with_context(question,context,provider,diagnostics_path=diagnostics_path)
    result['context_passage_ids']=[c['passage_id'] for c in context];result['omitted_passage_ids']=omitted
    return result
