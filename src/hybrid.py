"""Hybrid document ranking with chunk-level dense retrieval."""
from .retrieval import bm25, rrf


def rank(query, documents, encoder, store, mode='hybrid', depth=30, kind=None):
    docs = {d['id']: d['text'] for d in documents if kind is None or d['kind'] == kind}
    lexical = bm25(query, docs, k=depth) if mode != 'dense' else []
    dense = []
    if mode != 'bm25':
        points = store.search(encoder.encode([query], query=True)[0], limit=max(depth * 5, 100), kind=kind)
        seen = set()
        for p in points:
            doc_id = p.payload['document_id']
            if doc_id in docs and doc_id not in seen:
                dense.append((doc_id, float(p.score)))
                seen.add(doc_id)
                if len(dense) == depth:
                    break
    if mode == 'bm25':
        return lexical
    if mode == 'dense':
        return dense
    return rrf([[d for d, _ in lexical], [d for d, _ in dense]])[:depth]
