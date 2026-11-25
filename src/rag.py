"""Source-reference checks. These do not determine whether a claim is supported."""

def verify_citation(citation, documents, chunks):
    doc = documents.get(citation.get('document_id'))
    chunk = chunks.get(citation.get('chunk_id'))
    quote = citation.get('quote')
    if doc is None or chunk is None:
        return False
    if chunk.get('document_id') != citation.get('document_id'):
        return False
    start, end = chunk['start'], chunk['end']
    if not (0 <= start < end <= len(doc)) or chunk['text'] != doc[start:end]:
        return False
    return isinstance(quote, str) and bool(quote.strip()) and quote in chunk['text']
