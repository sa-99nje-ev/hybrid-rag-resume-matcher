"""Dependency-free BM25 and reciprocal rank fusion at document granularity."""
import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"(?i)(?:\.net|[a-z0-9]+(?:[.+#/\-][a-z0-9+#]+)*[+#]*)")

def tokenize(text):
    return TOKEN.findall(text.casefold())


def bm25(query, documents, k=10, k1=1.5, b=0.75):
    """documents maps stable document IDs to text; ties break by ID."""
    if not documents or k <= 0:
        return []
    tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
    avg = sum(map(len, tokens.values())) / len(tokens)
    if not avg:
        return []
    q = set(tokenize(query))
    df = Counter(term for words in tokens.values() for term in set(words) if term in q)
    scores = []
    for doc_id, words in tokens.items():
        counts = Counter(words)
        score = 0.0
        for term in q:
            freq = counts[term]
            if freq:
                idf = math.log(1 + (len(tokens) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(words) / avg))
        if score > 0:
            scores.append((doc_id, score))
    return sorted(scores, key=lambda x: (-x[1], x[0]))[:k]


def rrf(rankings, k=60):
    if k < 0:
        raise ValueError('k must be nonnegative')
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, 1):
            if doc_id not in seen:
                scores[doc_id] += 1 / (k + rank)
                seen.add(doc_id)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
