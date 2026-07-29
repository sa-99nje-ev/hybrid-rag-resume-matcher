# 🎯 Intelligent Resume–Job Matching System

An intelligent multi-stage **Information Retrieval (IR)** and **Learning-to-Rank (LTR)** system designed to improve resume recommendation by combining lexical retrieval, semantic search, neural reranking, and machine learning-based ranking.

Unlike traditional keyword-based resume search, this project integrates hybrid retrieval and ranking techniques to better capture recruiter intent and candidate skill relevance.

---

## 📌 Overview

Recruiters often rely on keyword matching, which struggles to identify semantically relevant candidates.

This project addresses that challenge by building a **multi-stage ranking pipeline** that progressively refines candidate retrieval through:

- Query Expansion
- Hybrid Lexical + Semantic Retrieval
- Neural Reranking
- Learning-to-Rank Optimization

The final output is a ranked list of the **Top-K most relevant resumes** for a given job description.

---

# 🏗️ System Architecture

```
                   Job Description
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Query Expansion (Flan-T5)      │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ Hybrid Retrieval                │
        │                                 │
        │ • BM25 (Sparse Search)          │
        │ • BGE-Large + FAISS             │
        └─────────────────────────────────┘
                          │
                          ▼
            Reciprocal Rank Fusion (RRF)
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ BGE Cross Encoder               │
        │ Neural Re-ranking               │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ LightGBM LambdaRank             │
        │ Final Learning-to-Rank Stage    │
        └─────────────────────────────────┘
                          │
                          ▼
             Top-K Ranked Resume Results
```

---

# 🚀 Key Features

- Multi-stage hybrid retrieval pipeline
- Query expansion using Flan-T5
- Sparse retrieval using BM25
- Dense semantic retrieval using BGE-Large + FAISS
- Reciprocal Rank Fusion (RRF)
- Neural reranking using BGE Cross-Encoder
- Learning-to-Rank optimization using LightGBM LambdaRank
- Interactive Gradio interface
- Evaluation using standard Information Retrieval metrics

---

# 🛠️ Tech Stack

## Languages

- Python

## Retrieval

- BM25
- FAISS
- Reciprocal Rank Fusion (RRF)

## NLP Models

- Flan-T5
- BGE-Large Embeddings
- BGE Cross Encoder

## Machine Learning

- LightGBM LambdaRank
- Scikit-learn

## Libraries

- Hugging Face Transformers
- Sentence Transformers
- Pandas
- NumPy
- Gradio

---

# 📊 Learning-to-Rank Features

The LambdaRank model uses engineered relevance features including:

- BM25 score
- Dense embedding similarity
- Cross-Encoder relevance score
- Skill overlap score
- Job title alignment
- Query length normalization
- Resume length normalization

---

# 📈 Evaluation

The pipeline was evaluated using standard Information Retrieval metrics.

### Self-Retrieval Performance

| Metric | Score |
|---------|------:|
| Precision@1 | **0.94** |
| Recall@10 | **0.99** |
| NDCG@10 | **0.964** |
| MAP | **0.956** |
| MRR | **0.956** |

> These results demonstrate the effectiveness of the hybrid retrieval and ranking pipeline in retrieving highly relevant resumes from the corpus.

---

# 📂 Repository Structure

```
.
├── notebooks/
├── data/
├── models/
├── evaluation/
├── app/
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/sa-99nje-ev/intelligent-resume-matching-system.git

cd intelligent-resume-matching-system
```

Install dependencies

```bash
pip install torch transformers sentence-transformers
pip install faiss-cpu rank-bm25 lightgbm
pip install scikit-learn pandas numpy gradio tqdm
```

Run the notebook or launch the Gradio application.

---

# 🔮 Future Work

- Fine-tune retrieval models on domain-specific resume datasets
- Incorporate Graph Neural Networks for candidate relationship modeling
- Deploy as a scalable REST API
- Extend to multilingual resume matching
- Integrate LLM-based explanation generation for recruiter recommendations

---

# 👨‍💻 Author

**Sanjeev Rao**

GitHub: https://github.com/sa-99nje-ev

---

# 📄 License

This project is intended for academic and educational purposes.
