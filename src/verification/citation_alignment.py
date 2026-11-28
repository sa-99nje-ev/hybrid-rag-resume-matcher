"""Conservative local quotation-span alignment; no semantic entailment claim."""
import re, unicodedata
from difflib import SequenceMatcher

TOKENS=re.compile(r'\w+(?:[.,]\d+)*|[%$€£+\-/<>=]',re.UNICODE)

def normalized_with_map(text):
    chars=[]; mapping=[]; i=0; pending_space=None
    while i<len(text):
        start=i;i+=1
        while i<len(text) and unicodedata.combining(text[i]):i+=1
        part=unicodedata.normalize('NFC',text[start:i])
        for ch in part:
            if ch.isspace():
                if chars:pending_space=(start,i)
            else:
                if pending_space and chars:
                    chars.append(' ');mapping.append(pending_space);pending_space=None
                chars.append(ch);mapping.append((start,i))
    return ''.join(chars),mapping

def align_quote(passage_id,quote,evidence):
    result={'passage_id':passage_id,'original_quote':quote,'exact_span':None,'confidence':None,'status':'invalid'}
    if passage_id not in evidence:return {**result,'status':'invalid_passage_id'}
    source,mapping=normalized_with_map(evidence[passage_id]);target,_=normalized_with_map(quote)
    if not target:return {**result,'status':'empty_quote'}
    starts=[m.start() for m in re.finditer('(?='+re.escape(target)+')',source)]
    if len(starts)==1:
        start=starts[0];end=start+len(target)
        return {**result,'exact_span':evidence[passage_id][mapping[start][0]:mapping[end-1][1]],'confidence':1.0,'status':'exact'}
    if len(starts)>1:return {**result,'status':'ambiguous'}
    qt=[m.group().casefold() for m in TOKENS.finditer(target)];st=list(TOKENS.finditer(source))
    if len(qt)<4:return result
    candidates=[]
    for i in range(len(st)-len(qt)+1):
        chunk=st[i:i+len(qt)]
        if [m.group().casefold() for m in chunk]!=qt:continue
        start,end=chunk[0].start(),chunk[-1].end();span=source[start:end]
        score=SequenceMatcher(None,target.casefold().strip(' .,:;'),span.casefold(),autojunk=False).ratio()
        if score>=0.97:candidates.append((start,end,score))
    if len(candidates)!=1:return {**result,'status':'ambiguous' if len(candidates)>1 else 'invalid'}
    start,end,score=candidates[0]
    return {**result,'exact_span':evidence[passage_id][mapping[start][0]:mapping[end-1][1]],'confidence':score,'status':'aligned_punctuation_only'}
