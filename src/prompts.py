"""Prompt options; few-shot examples require independently verified source evidence."""
from .llm import Report

BASE = ('Answer only from supplied evidence. Return a JSON object with claims containing text, status and citations. '
        'Valid statuses: directly_supported, partially_supported, transferable, not_evidenced, insufficient_information. '
        'Citations contain document_id, chunk_id and exact quote. Do not infer absence of skill from absent evidence; '
        'never treat scores as hiring probabilities.')


def instruction(style):
    if style == 'zero': return BASE
    if style == 'stepwise':
        return BASE + (' Identify requirements, inspect supplied evidence, assess each requirement, identify what is not evidenced, '
                       'and return the final structured report. Do not expose private chain of thought.')
    if style == 'few': return BASE + ' Follow the independently reviewed examples supplied as data.'
    raise ValueError('Unknown prompt style')


def validate_examples(examples):
    if not examples: raise ValueError('Few-shot prompting requires independently verified examples')
    for example in examples:
        if example.get('human_verified') is not True: raise ValueError('Example lacks human verification')
        evidence={(x['document_id'],x['chunk_id']):x['text'] for x in example['evidence']}
        report=Report.model_validate(example['report'])
        for claim in report.claims:
            if claim.status in {'directly_supported','partially_supported','transferable'} and not claim.citations:
                raise ValueError('Supported example claim lacks a citation')
            for citation in claim.citations:
                if not citation.quote.strip() or citation.quote not in evidence.get((citation.document_id,citation.chunk_id),''):
                    raise ValueError('Example citation does not match evidence')
    return examples
