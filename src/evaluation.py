"""Metrics under incomplete judgments: avoid silently labeling unknowns negative."""
import math


def judged_metrics(ranked, judgments, k=10):
    if len(ranked) != len(set(ranked)):
        raise ValueError('Duplicate ranked document IDs')
    if any(not isinstance(v, (int, float)) or v < 0 for v in judgments.values()):
        raise ValueError('Relevance grades must be nonnegative numbers')
    top = ranked[:k]
    coverage = sum(doc in judgments for doc in top) / len(top) if top else 0.0
    # Only report conventional metrics when every returned document is judged.
    if any(doc not in judgments for doc in top):
        return {'judgment_coverage': coverage, 'metrics': None}
    relevant = {doc for doc, grade in judgments.items() if grade > 0}
    hits = sum(doc in relevant for doc in top)
    precision = hits / k if k else None
    recall = hits / len(relevant) if relevant else None
    rr = next((1 / (i + 1) for i, doc in enumerate(top) if doc in relevant), 0.0)
    dcg = sum((2 ** judgments[doc] - 1) / math.log2(i + 2) for i, doc in enumerate(top))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return {'judgment_coverage': coverage, 'metrics': {'precision': precision,
            'recall': recall, 'reciprocal_rank': rr, 'ndcg': dcg / idcg if idcg else None},
            'note': 'Recall and nDCG are relative to supplied judgments, not necessarily the whole corpus.'}
