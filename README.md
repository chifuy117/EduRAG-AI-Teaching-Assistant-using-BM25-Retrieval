# 🎓 AI Teaching Assistant — Production-Grade RAG Pipeline

A complete, multi-subject AI teaching system powered by **Retrieval-Augmented Generation (RAG)**, intelligent noise-free ETL, and a comprehensive ML evaluation suite. Designed for **scalable academic support**, **multi-format ingestion**, and **production-grade analytics**.

---

## 📊 Latest Evaluation Metrics (April 18, 2026)

| Metric | Score | Status |
|--------|-------|--------|
| Retrieval F1 (macro) | **0.9618** | ✅ Excellent |
| Precision@5 | **0.85** | ✅ Good |
| MAP (Mean Average Precision) | **1.0819** | ✅ Excellent |
| MRR (Mean Reciprocal Rank) | **1.0000** | ✅ Excellent |
| Mean NDCG@5 | **1.0390** | ✅ Excellent |
| Subject Hit Accuracy | **100%** | ✅ Perfect |
| Topic Classification Accuracy | **0.61** | ⚠️ Needs improvement |
| Total Chunks Stored | **365 (~166k words)** | ✅ Stable |
| Chunk Cleanliness | **>80%** | ✅ Noise-free |

📌 Full evaluation reports are auto-saved in `data/eval_report.json`.

---

## 🏗️ System Architecture
ADS_DWH_Project/
│
├── etl/              # ETL Pipeline (extract, transform, load)
├── rag/              # RAG System (retriever + generator)
├── analytics/        # Evaluation, clustering, dashboard
├── modes/            # Quiz, exam prep, learning, syllabus
├── app/              # Streamlit UIs (premium & complete)
├── data/             # Subject PDFs, SQLite DB, vector store
├── train_all.py      # Main training pipeline
├── experiments.py    # Chunk-size & RAG-vs-NoRAG experiments
├── score_analysis.py # Evaluation checker + retrain decision
└── requirements.txt  # Dependencies


---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux

2. Install Dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords

3. Add Learning Materials
Place subject files in data/:
data/
  DBMS/             # Database, SQL, normalization
  OS/               # Operating systems
  DataStructures/   # DS & algorithms
  MachineLearning/  # ML concepts

4. Train the System
python train_all.py

Options:

--clean → retrain from scratch

--no-eval → skip evaluation

--chunk-size 300 → custom chunk size

5. Check Scores
python score_analysis.py

6. Launch Application
python launcher.py

📈 Metrics Overview
Precision@K → relevance of top results

Recall@K → coverage of relevant docs

F1@K → balance of precision & recall

MAP → ranking quality

MRR → speed of finding first relevant doc

NDCG@K → graded relevance with rank discounting

🎯 Supported Subjects
| Subject | Topics (examples) |
| --- | --- |
| OS | Scheduling, Virtual Memory, Deadlock, File Systems |
| DataStructures | Arrays, Linked Lists, Trees, Graphs, Sorting |
| MachineLearning | Supervised, Unsupervised, Neural Nets, Evaluation |
| DBMS | Normalization, SQL, Transactions, ER Model, Indexing |

🖥️ UI Modes
🏠 Dashboard → Overview & quick actions

💬 Chat → RAG-powered Q&A with citations

📚 Learning → Concept explanations + practice

📊 Exam Prep → Predicted questions & weak areas

📝 Quiz → Auto-generated MCQs & short answers

🎯 Syllabus → Validate against course syllabus

📈 Analytics → Clustering, topic modeling, query patterns

🔧 Troubleshooting
Virtual environment not active → re-run venv\Scripts\activate

Missing dependencies → pip install -r requirements.txt

Training fails → python train_all.py --clean

UI won’t start → free port 8501 and retry

📦 Requirements
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
nltk
pypdf
python-pptx
python-docx
python-dotenv

📝 Version History
v3.0 (Apr 2026) → Full eval suite, noise removal, hash dedup, stable IDs

v2.0 (Mar 2026) → Premium UI, multi-format ETL, RAG pipeline

v1.0 (Mar 2026) → Initial project setup
