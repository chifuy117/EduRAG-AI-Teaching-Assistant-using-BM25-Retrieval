# 🎓 AI Teaching Assistant — Production-Grade RAG Pipeline

> A complete, multi-subject AI teaching system powered by **Retrieval-Augmented Generation (RAG)**, intelligent noise-free ETL, and a comprehensive ML evaluation suite.

---

## 📊 Current Evaluation Metrics (Latest Run — April 18, 2026)

| Metric | Score | Status |
|--------|-------|--------|
| **Retrieval F1 (macro)** | **0.9618** | ✅ Excellent |
| **Precision@5** | **0.85** | ✅ Good |
| **MAP** (Mean Average Precision) | **1.0819** | ✅ Excellent |
| **MRR** (Mean Reciprocal Rank) | **1.0000** | ✅ Excellent |
| **Mean NDCG@5** | **1.0390** | ✅ Excellent |
| **Subject Hit Accuracy** | **1.0 (100%)** | ✅ Perfect |
| **Topic Classification Acc** | **0.61** | ✅ Needs improvement |
| **Total Chunks Stored** | **365** | ✅ 166 k words |
| **Chunk Cleanliness** | **>80%** | ✅ Noise-free |

> Full report saved automatically after each training run → `data/eval_report.json`

---

## 🏗️ Project Architecture

```
ADS_DWH_Project/
│
├── 📥 etl/                        # ETL Pipeline
│   ├── extract.py                 # Multi-format extractor (PDF, PPTX, DOCX, TXT, MD)
│   ├── transform.py               # Noise removal, dedup, hash IDs, quality filter
│   └── load.py                    # SQLite DWH + BM25 keyword vector store
│
├── 🤖 rag/                        # RAG System
│   ├── retriever.py               # BM25-based retriever with full metrics
│   └── generator.py               # Answer generator with quality scoring
│
├── 📊 analytics/                  # Analytics & Evaluation
│   ├── evaluator.py               # NEW: Full ML eval suite (F1, MAP, NDCG, confusion matrix)
│   ├── clustering.py              # K-Means clustering, LDA, association rules
│   └── dashboard.py               # Streamlit analytics dashboard
│
├── 🎯 modes/                      # Learning Modes
│   ├── quiz.py                    # Quiz generator
│   ├── exam.py                    # Exam preparation mode
│   ├── learning.py                # Concept explanation mode
│   └── syllabus.py                # Syllabus validator
│
├── 🖥️ app/
│   ├── ui_premium.py              # Premium Streamlit UI (8 modes)
│   └── ui_complete.py             # Alternate complete UI
│
├── 🗄️ data/
│   ├── DBMS/                      # DBMS subject PDFs → 52 chunks
│   ├── OS/                        # OS subject PDFs   → 167 chunks
│   ├── DataStructures/            # DS PDFs           → 80 chunks       
│   ├── MachineLearning/           # ML PDFs           → 65 chunks
│   ├── warehouse.db               # SQLite (chunks, queries, metrics tables)
│   ├── vector_db/                 # Keyword-inverted index (JSON)
│   ├── eval_report.json           # Latest evaluation report
│   └── training_log.json          # Latest training log
│
├── train_all.py                   # 🚀 Main training pipeline (7 steps)
├── experiments.py                 # Chunk-size & RAG-vs-NoRAG experiments
├── check_imports.py               # Module connectivity check
├── score_analysis.py              # Score checker + retrain decision
├── main.py                        # CLI entry point
├── launcher.py                    # Interactive launcher
├── config.py                      # Configuration
├── syllabus.py                    # Subject syllabus definitions
└── requirements.txt               # Python dependencies
```

---

## 🚀 Quick Start (Step-by-Step)

### Step 1: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Verify activation (should show venv path)
where python  # Windows
which python  # macOS/Linux
```

### Step 2: Install Dependencies
```bash
# Activate venv first (if not already)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install Python packages
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords
```

### Step 3: Add Your Learning Materials
Place PDFs/PPTX/DOCX/TXT/MD files into subject-named folders inside `data/`:
```
data/
  DBMS/          → database, SQL, normalization materials
  OS/            → operating system notes
  DataStructures/→ DS & algorithms slides
  MachineLearning/→ ML algorithms and concepts materials
```

### Step 4: Train the System
```bash
# Activate venv first
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Run full training pipeline
python train_all.py

# Alternative options:
python train_all.py --clean           # Clean previous data first, then retrain
python train_all.py --no-eval         # Skip evaluation step (faster)
python train_all.py --chunk-size 300  # Custom chunk size
```

### Step 5: Check Training Scores
```bash
# Activate venv first
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Check evaluation results
python score_analysis.py

# Optional: Re-run evaluation
python analytics/evaluator.py --k 5
```

### Step 6: Launch the Application
```bash
# Activate venv first
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Launch interactive menu
python launcher.py

# Or launch specific UI directly:
streamlit run app/ui_premium.py    # Premium UI (recommended)
streamlit run app/ui_complete.py   # Complete UI
streamlit run app/ui.py            # Basic chat UI
```

### Step 7: Start Using the System
1. Open browser at `http://localhost:8501`
2. Select subject from dropdown
3. Ask questions or generate quizzes
4. Explore analytics and syllabus features

---

## � Daily Usage (After Initial Setup)

Once trained, you can run the system anytime:

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Launch the application
python launcher.py

# Or check system status
python main.py --info

# Or test RAG system
python main.py --test-rag
```

### Adding New Materials
To add new PDFs or update content:
```bash
# Activate venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Retrain with new data
python train_all.py --clean

# Check updated scores
python score_analysis.py
```

---
## 🐛 Troubleshooting

### Common Issues

**Virtual Environment Not Activated**
```bash
# Check if venv is active
where python  # Should show venv path
# If not, activate:
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt
# Install specific package
pip install plotly  # or any missing package
```

**Training Fails**
```bash
# Clean and retrain
python train_all.py --clean
# Check data directory
ls data/  # Should have subject folders with PDFs
```

**UI Won't Start**
```bash
# Check port availability
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # macOS/Linux
# Kill process if needed, then restart
```

**Low Evaluation Scores**
```bash
# Add more high-quality PDFs
# Retrain with different chunk size
python train_all.py --chunk-size 300 --clean
```

---
## �🔧 Full Pipeline Walkthrough (train_all.py)

| Step | Module | What Happens |
|------|--------|--------------|
| 1 | Environment | Dependency check, NLTK resources |
| 2 | Extract | `etl/extract.py` reads all files from `data/` |
| 3 | Transform | `etl/transform.py` cleans noise, chunks, assigns stable hash IDs |
| 4 | Junk Delete | Classifies & removes low-quality files |
| 5 | Load SQL | `etl/load.py` → INSERT OR IGNORE into `chunks` table |
| 6 | Load Vec | BM25 keyword index built & serialised to `data/vector_db/` |
| 7 | **Evaluate** | `analytics/evaluator.py` → Precision, Recall, F1, MAP, NDCG written to `evaluation_metrics` table **+ JSON** |

---

## 📈 Metrics Explained

### Retrieval Metrics (averaged over 8 test queries)
| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision@K** | TP / K | Fraction of retrieved docs that are relevant |
| **Recall@K** | TP / Total Relevant | Fraction of relevant docs retrieved |
| **F1@K** | 2·P·R / (P+R) | Harmonic mean of Precision & Recall |
| **MAP** | Mean of Avg Precision over all queries | Quality of ordering |
| **MRR** | Mean of 1/rank of first relevant | How quickly we find first relevant doc |
| **NDCG@K** | DCG / IDCG | Graded relevance with rank discounting |

### Classification Metrics (topic assignment, 200 samples)
| Metric | Formula |
|--------|---------|
| **Overall Accuracy** | Correct / Total |
| **Macro Precision** | Average precision per class |
| **Macro Recall** | Average recall per class |
| **Macro F1** | Average F1 per class |
| **Confusion Matrix** | Full per-class TP/FP/FN/TN |

---

## 🧹 Data Quality Pipeline

### Noise Removal (etl/transform.py)
- ✅ Strips PDF page numbers (`Page 5 of 12`)
- ✅ Removes URLs, emails, LaTeX commands
- ✅ Removes non-ASCII / corrupted characters
- ✅ **Preserves technical numbers** (3NF, IPv4, O(n²), HTTP/2)
- ✅ Domain-aware stopword removal (keeps CS acronyms: OS, AI, IT, IO)

### Deduplication
- ✅ Content-hash (MD5) based chunk IDs → `chunk_<12hexchars>`
- ✅ SQLite `content_hash UNIQUE` column → `INSERT OR IGNORE`
- ✅ Vector store dedup by `chunk_id`
- ✅ **Idempotent**: re-running training never duplicates data

### Quality Filter
- ✅ Minimum 20 words per chunk (garbage chunks discarded)
- ✅ Quality score on generated answers (keyword overlap + completeness)

---

## 🎯 Supported Subjects & Topics

| Subject | Chunks | Sample Topics |
|---------|--------|---------------|
| **OS** | 121 | Process scheduling, Virtual memory, Deadlock, Synchronisation, File systems |
| **DataStructures** | 62 | Arrays, Linked lists, Trees, Graphs, Sorting |
| **MachineLearning** | 52 | Supervised, Unsupervised, Evaluation, Neural nets |
| **DBMS** | 38 | Normalization, SQL, Transactions, ER model, Indexing |

---

## 🔬 Running Experiments

```bash
# Compare chunk sizes (200, 500, 1000 words)
python experiments.py

# Produces:
#   data/experiments_results.json
#   data/experiment_chunk_size.png
#   data/experiment_rag_vs_no_rag.png
```

---

## 🛠️ Module Connections

```
train_all.py
  └→ etl.extract.DataExtractor
  └→ etl.transform.DataTransformer
  └→ etl.load.DataWarehouse ←──────────────────────────────┐
       └→ data/warehouse.db (chunks, queries, eval_metrics) │
       └→ data/vector_db/chunks.json + index.json           │
  └→ analytics.evaluator.Evaluator ───────────────────────→─┘
       └→ rag.retriever.DocumentRetriever
       └→ ClassificationMetrics
       └→ RetrievalMetrics (P, R, F1, MAP, MRR, NDCG)
       └→ QualityMetrics (keyword overlap, completeness)

app/ui_premium.py
  └→ etl.load.DataWarehouse
  └→ rag.retriever.DocumentRetriever
  └→ rag.generator.AnswerGenerator
  └→ modes.learning.LearningMode
  └→ modes.exam.ExamMode
  └→ modes.quiz.QuizGenerator
  └→ modes.syllabus.SyllabusValidator
  └→ syllabus.get_subject_from_query
```

---

## ⚡ When to Retrain

Run `python score_analysis.py` after any new PDFs are added. The script will automatically flag retraining if:

- Retrieval F1 < 0.80
- Precision@5 < 0.70
- Classification Accuracy < 0.50
- Chunk cleanliness < 80%
- Total chunks < 100
- MAP < 0.70

**Current status: ✅ No retraining needed** (F1=0.93, NDCG=0.99, Subject Accuracy=100%)

To force retrain from scratch:
```bash
python train_all.py --clean
```

---

## 📋 Database Schema

### `chunks` table
```sql
chunk_id       TEXT PRIMARY KEY   -- stable hash ID: chunk_<md5[:12]>
content_hash   TEXT UNIQUE        -- dedup key
subject        TEXT               -- DBMS | OS | DataStructures | MachineLearning
topic          TEXT               -- normalization | process_scheduling | ...
difficulty     TEXT               -- easy | medium | hard
content        TEXT               -- cleaned chunk text
word_count     INTEGER
source_file    TEXT
```

### `evaluation_metrics` table
```sql
run_id          TEXT PRIMARY KEY
run_timestamp   TIMESTAMP
subject         TEXT
precision_at_k  REAL
recall          REAL
f1_score        REAL
accuracy        REAL
map_score       REAL
ndcg_score      REAL
total_chunks    INTEGER
notes           TEXT
```

---

## 🖥️ UI Modes

| Mode | Description |
|------|-------------|
| 🏠 Dashboard | Overview, progress, quick actions |
| 💬 Chat Mode | RAG-powered Q&A with source citations |
| 📚 Learning Mode | Concept explanations + examples + practice |
| 📊 Exam Prep | Predicted questions, weak areas, study schedule |
| 📝 Quiz | Auto-generated MCQ/short-answer quizzes |
| 📈 Analytics | Clustering, topic modeling, query patterns |
| 🎯 Syllabus | Validate topics against course syllabus |
| ❓ Help | Documentation and system info |

---

## 📦 Requirements

```
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
```

Install: `pip install -r requirements.txt`

---

## 📝 Versions

| Version | Date | Changes |
|---------|------|---------|
| v3.0 | Apr 2026 | Full eval suite (F1, MAP, NDCG), noise removal, hash dedup, stable IDs |
| v2.0 | Mar 2026 | Premium UI, multi-format ETL, RAG pipeline |
| v1.0 | Mar 2026 | Initial project setup |#   E d u R A G - A I - T e a c h i n g - A s s i s t a n t - u s i n g - B M 2 5 - R e t r i e v a l  
 