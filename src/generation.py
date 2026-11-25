"""Evidence context and structured generation. Citation checks do not prove entailment."""
from .rag import verify_citation
from .llm import Report


def build_context(ranking, chunks, question="", max_chars=12000):
    order = {doc_id: i for i, (doc_id, _) in enumerate(ranking)}
    from .retrieval import bm25
    candidate_chunks = [x for x in chunks if x['document_id'] in order]
    relevance = dict(bm25(question, {x['chunk_id']:x['text'] for x in candidate_chunks}, k=len(candidate_chunks)))
    chosen, seen, size = [], set(), 0
    for c in sorted(candidate_chunks, key=lambda x: (order[x['document_id']], -relevance.get(x['chunk_id'], 0), x['start'])):
        if c['text'] in seen or size + len(c['text']) > max_chars:
            continue
        chosen.append({k:c[k] for k in ('document_id','chunk_id','text')})
        seen.add(c['text']); size += len(c['text'])
    return chosen


def answer(question, ranking, documents, chunks, llm, prompt_style='zero', examples=None):
    context = build_context(ranking, chunks, question)
    from .prompts import instruction, validate_examples
    data = {'question':question,'evidence':context}
    step_analysis = None
    step_stats = None
    if prompt_style == 'stepwise':
        step_analysis, step_stats = llm.expand(question)
        data['extracted_requirements'] = step_analysis.model_dump()
    if prompt_style == 'few': data['verified_examples'] = validate_examples(examples)
    report, stats = llm.generate(instruction(prompt_style), data, Report)
    docs = {d['id']:d['text'] for d in documents}
    chunkmap = {c['chunk_id']:c for c in chunks}
    provided = {c['chunk_id'] for c in context}
    errors = []
    if not report.claims: errors.append('No claims returned')
    allowed = {'directly_supported','partially_supported','transferable','not_evidenced','insufficient_information'}
    for i, claim in enumerate(report.claims):
        if claim.status not in allowed: errors.append(f'claim {i}: unknown status')
        if claim.status in {'directly_supported','partially_supported','transferable'} and not claim.citations: errors.append(f'claim {i}: missing citation')
        for citation in claim.citations:
            if citation.chunk_id not in provided or not verify_citation(citation.model_dump(), docs, chunkmap): errors.append(f'claim {i}: invalid citation')
    steps = None
    if prompt_style == 'stepwise':
        steps = {'requirement_extraction':step_analysis.model_dump(), 'evidence_chunk_ids':[x['chunk_id'] for x in context],
                 'not_evidenced_or_uncertain_claims':[i for i,c in enumerate(report.claims) if c.status in {'not_evidenced','insufficient_information'}],
                 'extraction_stats':step_stats}
    return {'report':report.model_dump(), 'steps':steps, 'citation_errors':errors, 'source_checks_passed':not errors,
            'semantic_support': 'not automatically verified', 'stats':stats}
