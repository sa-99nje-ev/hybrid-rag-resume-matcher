# Advanced Hybrid RAG Resume–Job Intelligence

A modular Python project for searching resume and job documents and generating evidence-linked reports. It combines BM25, BGE embeddings, persistent Qdrant retrieval, reciprocal rank fusion, a BGE cross-encoder, and Groq-assisted generation. The system supports human decision-making; ranking scores are not hiring probabilities, and missing evidence does not prove that a person lacks a skill.

## Architecture

1. Preserve source documents and split them into sections and passages with stable identifiers and source offsets.
2. Encode passages with `BAAI/bge-small-en-v1.5`; use CUDA when available, fall back to CPU, and cache normalized embeddings locally.
3. Search passages in Qdrant and documents with BM25, aggregate dense hits to documents, and fuse rankings with reciprocal rank fusion. FAISS provides an exact-search reference for indexing experiments.
4. Optionally rerank retrieved candidates using `BAAI/bge-reranker-base`.
5. Optionally use Groq for query expansion, structured evidence-linked generation, or a bounded read-only tool-calling workflow. Citation checks verify source identifiers and exact quotations; they do not prove semantic claim support.

The CLI provides indexing, search, question answering, and an agent. Search supports BM25, dense, and hybrid modes. Question answering provides naive, hybrid, and advanced configurations. Zero-shot and stepwise prompts are implemented; few-shot prompts require independently human-verified examples with matching evidence.

## Setup

Use Python on Windows and install the listed dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Select a PyTorch build compatible with your NVIDIA driver, or use CPU. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Start Qdrant using `docker compose up -d` when Docker is available. Qdrant server storage is persistent; embedded local mode must not be assumed to behave like server HNSW indexing.

Copy `.env.example` to `.env` and configure the required environment variables. Keep credentials out of version control. `GROQ_MODEL` and `QDRANT_URL` are configurable. Model artifacts and embedding caches remain local.

## Data and privacy

No resumes, job descriptions, annotation worksheets, dataset files, model weights, vector storage, or credentials are distributed in this repository. Bring source data only after verifying its license and privacy requirements. The JSONL corpus format contains a unique `id`, `kind` (`resume` or `job`), and original `text` per document. Dataset adapters are provided in `scripts/`; preserve original files and provenance separately.

The larger local experiments used the researchers' [Djinni recruitment corpus](https://aclanthology.org/2024.unlp-1.2/), published in English and Ukrainian partitions under MIT-labeled terms. The authors identify residual personal or protected attributes. A smaller separate research dataset is also documented in [DATASET_AUDIT.md](DATASET_AUDIT.md). Dataset availability or redaction does not by itself establish permission to transmit personal data to Groq. Review rights and privacy before external processing. The `--allow-external-data` option enables a request; it does not grant permission or anonymize data.

## CLI

With a locally prepared corpus and Qdrant server:

```powershell
python main.py --corpus data\processed\documents.jsonl --collection my_documents index
python main.py --corpus data\processed\documents.jsonl --collection my_documents search "Python developer" --kind resume --mode hybrid
python main.py --corpus data\processed\documents.jsonl --collection my_documents ask "Your question" --mode advanced --allow-external-data
python main.py --corpus data\processed\documents.jsonl --collection my_documents agent "Your question" --allow-external-data
```

The indexer upserts stable chunk IDs but does not delete obsolete chunks. Use a new collection when the corpus or chunking changes. Treat source documents as untrusted data. Inspect outputs and citations before relying on them.

## Verified infrastructure and evaluation status

A local, fixed-seed Djinni sample of 10,000 candidate profiles and 200 jobs produced 19,988 chunks. On 200 job queries, Qdrant's nearest-neighbor agreement with FAISS exact search was 0.9990 at 10 and 0.9995 at 20. A separate filtered query incremented Qdrant's HNSW graph-search telemetry. These are infrastructure checks on one sample, not resume–job relevance or hiring accuracy; other queries may use different search plans. The sampled and earlier source-ordered runs overlap and are not independent validation splits.

An RTX 4050 Laptop GPU was verified with a CUDA-enabled PyTorch environment. Naive, hybrid, advanced, and stepwise RAG and the bounded agent were demonstrated on a short public-domain U.S. Bureau of Labor Statistics passage. Source-reference checks passed on the recorded demonstrations; this is not an independent generation-quality evaluation.

Independent human relevance judgments, leakage-safe validation and test splits, controlled semantic retrieval ablations, human-assessed RAG and prompt quality, and verified few-shot examples remain pending. Do not treat unjudged pairs as irrelevant, derive labels from embeddings, or use the same model as the sole judge of its answers. [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) describes the local human-review workflow. The experimental notebook retains its existing code and explanations; local data and result files referenced by some cells are intentionally excluded and must be prepared separately.
