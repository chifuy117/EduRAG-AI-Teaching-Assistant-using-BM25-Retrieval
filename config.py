# Configuration File
# This file contains all configuration settings for the AI Teaching Assistant

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Database Configuration
DATABASE_PATH = "data/warehouse.db"
VECTOR_DB_PATH = "data/vector_db"

# Data Directories
DATA_DIR = "data"
PDF_DIR = "data"  # Root directory containing subject folders

# Processing Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_RETRIEVED_CHUNKS = 5

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.3
MAX_TOKENS = 1000



# Subjects Configuration
SUPPORTED_SUBJECTS = [
    "DBMS",
    "OS",
    "DataStructures",
    "Programming",
    "Algorithms",
    "Networks",
    "General"
]

# Analytics Configuration
ENABLE_ANALYTICS = True
LOG_QUERIES = True

# UI Configuration
PAGE_TITLE = "AI Teaching Assistant"
PAGE_ICON = "🎓"
LAYOUT = "wide"

# Evaluation Metrics
EVALUATION_METRICS = [
    "precision_at_k",
    "recall",
    "relevance_score",
    "answer_length",
    "context_relevance"
]

# File Extensions (multi-format support)
SUPPORTED_EXTENSIONS = [".pdf", ".pptx", ".docx", ".txt", ".md"]

# Chunking Configuration
TEXT_SPLITTER_CONFIG = {
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "separators": ["\n\n", "\n", " ", ""]
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    "index_type": "faiss",
    "metric": "cosine",
    "normalize_embeddings": True
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/app.log"

# Cache Configuration
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 1000

# Security Configuration
MAX_QUERY_LENGTH = 1000
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour