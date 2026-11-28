# Financial Forensics & Evidence Intelligence

A research system for retrieving financial filing evidence and checking numerical calculations. It combines lexical and semantic search with source citations and local Decimal arithmetic. Outputs require independent review and are not financial advice.

## Architecture

1. Ingest authorized financial documents as JSONL records with `id`, `kind` set to `financial`, and `text`; section-aware chunking preserves source offsets.
2. Retrieve with BM25 and compact BGE embeddings in persistent Qdrant, fuse rankings with reciprocal rank fusion, and rerank with the BGE cross-encoder. FAISS exact search remains available for infrastructure comparisons.
3. Send the selected passages and question in one structured Groq request for a concise evidence plan, source quotations, extracted facts and an arithmetic expression. Document text is treated as untrusted input.
4. Locally align quotations, check source values, fiscal periods and compatible units, and execute a restricted Decimal AST expression. Python constructs the numerical answer or returns a structured abstention. Original and verified quotations remain separately auditable.

Citation matching and arithmetic checks do not prove that a claim follows from a passage. Qualitative requests return cited evidence for review and abstain from a factual answer when semantic support cannot be verified. The numeric path is experimental and has not undergone independent accuracy evaluation.

## Data, provenance and privacy

The independently checked [FinRank retrieval evaluation](FINRANK_EVALUATION.md) uses the official [FinRank release](https://github.com/datanxt/FinRank) by Mansouri and colleagues, revision `916dc27e77b90e294ea1b3a2cc5b0abcf07392d1`, under CC BY-NC 4.0 with attribution. It contains curated SEC-derived passages and question-specific judgments; it is not a complete filings corpus. Respect source rights and third-party processing terms before sending documents to a hosted API. Reference answers remain local.

Data, raw passages, diagnostics, credentials, model weights, embedding caches and Qdrant storage are excluded from Git. The `--allow-external-data` flag permits an API request but does not grant data rights or anonymize content. Do not commit `.env` files.

## Verified retrieval results

The completed evaluation used 118 leakage-filtered test questions. The following metrics rank each question's gold passages against its curated hard negatives; gold evidence is therefore present by construction. They do not measure ordinary end-to-end search accuracy.

| Retrieval method | MRR | nDCG@10 |
|---|---:|---:|
| BM25 | 0.7756 | 0.8166 |
| BGE dense | 0.7785 | 0.8181 |
| Hybrid RRF | 0.7965 | 0.8281 |
| Hybrid RRF + cross-encoder | 0.8558 | 0.8705 |

The Qdrant collection held 5,230 indexed passages. Global-pool relevance judgments are incomplete; unjudged passages are not negatives. One question shared between train and test was excluded, but passage overlap across splits remains. Details, aggregate results and methodological caveats are in [FINRANK_EVALUATION.md](FINRANK_EVALUATION.md).

## Installation and configuration

Use Python in a virtual environment and install `requirements.txt`. Install a PyTorch build compatible with the local GPU or use CPU; see the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Start a persistent Qdrant server using `docker compose up -d`, or configure the existing server. Use the existing BGE model cache when available; no model download is required for the offline tests.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:FINFORENSICS_ENV_FILE = 'C:\secure\financial.env'
$env:FINFORENSICS_MODEL_CACHE = 'C:\secure\model-cache'
```

The optional external environment file can provide `HF_TOKEN`, `GROQ_API_KEY`, `GROQ_MODEL` and standard environment settings. Set `FINFORENSICS_ENV_FILE` to its existing path before starting the program. `FINFORENSICS_MODEL_CACHE` points to an existing Hugging Face cache; otherwise the standard cache is used. Environment values take precedence over `.env` files.

## Commands

With an authorized local corpus and running Qdrant:

```powershell
python main.py --corpus data/processed/financial_documents.jsonl index
python main.py --corpus data/processed/financial_documents.jsonl search "What changed in revenue?"
python main.py --corpus data/processed/financial_documents.jsonl ask "What changed in revenue between 2022 and 2023?" --allow-external-data
```

Indexing upserts current passage identifiers and does not remove obsolete records; use a separate collection when the corpus changes. The FinRank preparation, indexing and retrieval evaluation utilities are in `scripts/`; they require locally acquired data and should not overwrite existing experiment outputs. The held-out generation split is not enabled. The optional diagnostic path must be inside the ignored `results/` directory.

## Tests

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -B -m pytest -p no:cacheprovider -q
```

The tests use local fixtures and mocked provider responses. The evidence-plan provider accepts at most one request per question; retries are disabled for this workflow.

## Limitations and future work

Citation alignment, rate limiting and numerical verification have local test coverage but insufficient live validation. Numerical answer correctness, factual grounding, calibration and held-out generation results remain unverified. A later authorized study should complete development validation with independent judgments before any frozen held-out evaluation; do not interpret source-reference validity or ANN agreement as semantic accuracy.
