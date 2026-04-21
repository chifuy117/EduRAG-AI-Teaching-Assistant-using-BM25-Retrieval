# ETL - Load Module (v2 — Stable IDs, Deduplication, Metrics Table)
# Loads transformed data into SQL warehouse + keyword-indexed vector store

import os
import json
import sqlite3
import hashlib
from typing import List
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Document shim (keep consistent with rest of pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ─────────────────────────────────────────────────────────────────────────────
# DataWarehouse
# ─────────────────────────────────────────────────────────────────────────────

class DataWarehouse:
    """
    Manages two storage layers:
      1. SQLite  — chunks, queries, subjects, evaluation_metrics tables
      2. JSON "vector store" — keyword-indexed in-memory + disk store
    """

    def __init__(self,
                 db_path: str = "data/warehouse.db",
                 vector_db_path: str = "data/vector_db"):
        self.db_path = db_path
        self.vector_db_path = vector_db_path

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)

        # In-memory vector store
        self.chunks_data: List[dict] = []
        self.chunk_index: dict = {}          # word → [chunk_id, ...]

        self.init_sql_db()

    # ── Schema ───────────────────────────────────────────────────────────────

    def init_sql_db(self):
        """Create all tables (idempotent)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Main chunks table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                doc_id          TEXT,
                subject         TEXT,
                topic           TEXT,
                difficulty      TEXT,
                content         TEXT,
                content_hash    TEXT UNIQUE,
                word_count      INTEGER,
                source          TEXT,
                source_file     TEXT,
                source_type     TEXT,
                metadata        TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Query log (for analytics)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                query_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT,
                query_text      TEXT,
                retrieved_chunks TEXT,
                response        TEXT,
                relevance_score REAL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Subjects catalogue
        cur.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                subject_id  TEXT PRIMARY KEY,
                name        TEXT,
                description TEXT
            )
        ''')

        # Evaluation metrics (written after training / eval runs)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                run_id          TEXT PRIMARY KEY,
                run_timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                subject         TEXT,
                precision_at_k  REAL,
                recall          REAL,
                f1_score        REAL,
                accuracy        REAL,
                map_score       REAL,
                ndcg_score      REAL,
                total_chunks    INTEGER,
                notes           TEXT
            )
        ''')

        conn.commit()
        conn.close()

    # ── SQL Load ─────────────────────────────────────────────────────────────

    def load_to_sql(self, chunks: List[Document]) -> int:
        """
        Insert chunks into SQLite.  Uses INSERT OR IGNORE so re-runs are
        idempotent (content_hash is the dedup key).
        Returns number of newly inserted rows.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        inserted = 0
        for chunk in chunks:
            m = chunk.metadata
            content_hash = m.get('content_hash') or \
                           hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]

            cur.execute('''
                INSERT OR IGNORE INTO chunks
                (chunk_id, doc_id, subject, topic, difficulty,
                 content, content_hash, word_count, source,
                 source_file, source_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                m.get('chunk_id', f"chunk_{content_hash}"),
                m.get('doc_id', ''),
                m.get('subject', 'general'),
                m.get('topic', 'general'),
                m.get('difficulty', 'medium'),
                chunk.page_content,
                content_hash,
                m.get('word_count', len(chunk.page_content.split())),
                m.get('source', 'unknown'),
                m.get('source_file', 'unknown'),
                m.get('source_type', 'unknown'),
                json.dumps(m),
            ))
            if cur.rowcount:
                inserted += 1

        conn.commit()
        conn.close()
        print(f"  [Load SQL] Inserted {inserted} new chunks "
              f"({len(chunks) - inserted} already existed — skipped)")
        return inserted

    # ── Vector / Keyword Store ────────────────────────────────────────────────

    def load_to_vector_db(self, chunks: List[Document]) -> bool:
        """
        Build a keyword-inverted index and dump both the flat chunk list
        and the index to JSON files.  Existing disk data is merged
        (dedup by chunk_id) so re-runs don't double-store.
        """
        # Load existing data from disk first
        self._load_from_disk()

        existing_ids = {c['id'] for c in self.chunks_data}
        added = 0

        for chunk in chunks:
            chunk_id = chunk.metadata.get('chunk_id', '')
            if not chunk_id:
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
                chunk_id = f"chunk_{content_hash}"

            if chunk_id in existing_ids:
                continue  # Already indexed

            content_lower = chunk.page_content.lower()
            for word in content_lower.split():
                if len(word) >= 3:
                    self.chunk_index.setdefault(word, [])
                    if chunk_id not in self.chunk_index[word]:
                        self.chunk_index[word].append(chunk_id)

            self.chunks_data.append({
                'id':       chunk_id,
                'content':  chunk.page_content,
                'metadata': chunk.metadata,
            })
            existing_ids.add(chunk_id)
            added += 1

        # Persist to disk
        chunks_path = os.path.join(self.vector_db_path, "chunks.json")
        index_path  = os.path.join(self.vector_db_path, "index.json")

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_data, f, ensure_ascii=False)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_index, f, ensure_ascii=False)

        print(f"  [Load Vec] Added {added} new chunks to vector store "
              f"(total {len(self.chunks_data)})")
        return True

    def _load_from_disk(self):
        """Load existing vector store from disk into memory."""
        chunks_path = os.path.join(self.vector_db_path, "chunks.json")
        index_path  = os.path.join(self.vector_db_path, "index.json")

        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks_data = json.load(f)
            except Exception:
                self.chunks_data = []

        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.chunk_index = json.load(f)
            except Exception:
                self.chunk_index = {}

    # ── Combined ─────────────────────────────────────────────────────────────

    def load_all(self, chunks: List[Document]):
        """Load to both SQL and vector store."""
        print("  [Load] Starting data loading...")
        self.load_to_sql(chunks)
        self.load_to_vector_db(chunks)
        print("  [Load] Done.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_vectorstore(self) -> bool:
        """Load vector store from disk into memory. Returns True if data exists."""
        self._load_from_disk()
        return len(self.chunks_data) > 0

    def search_similar(self, query: str, k: int = 3) -> List[tuple]:
        """
        BM25-inspired keyword scoring:
          Score = Σ (term frequency in chunk) / log(1 + doc_freq)
        Returns list of (SimpleDoc, normalised_score).
        """
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) >= 2]

        if not query_words:
            return []

        chunk_scores: dict = {}

        for word in query_words:
            if word in self.chunk_index:
                doc_freq = len(self.chunk_index[word]) + 1  # smoothing
                import math
                idf_weight = 1.0 / math.log(1 + doc_freq)

                for chunk_id in self.chunk_index[word]:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + idf_weight

        # Normalise
        if not chunk_scores:
            return []

        max_score = max(chunk_scores.values())

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        id_to_chunk = {c['id']: c for c in self.chunks_data}

        for chunk_id, raw_score in sorted_chunks[:k]:
            chunk_data = id_to_chunk.get(chunk_id)
            if chunk_data is None:
                continue

            class _SimpleDoc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata

            doc = _SimpleDoc(chunk_data['content'], chunk_data['metadata'])
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append((doc, norm_score))

        return results

    # ── SQL Helpers ──────────────────────────────────────────────────────────

    def query_sql(self, sql: str, params: tuple = ()) -> List[tuple]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        return rows

    def log_query(self, user_id: str, query_text: str,
                  retrieved_chunks, response: str,
                  relevance_score: float = None):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO queries (user_id, query_text, retrieved_chunks, response, relevance_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, query_text, json.dumps(retrieved_chunks), response, relevance_score))
        conn.commit()
        conn.close()

    def save_evaluation_metrics(self, metrics: dict):
        """Persist an evaluation run's metrics to the DB."""
        import uuid
        run_id = str(uuid.uuid4())[:8]
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO evaluation_metrics
            (run_id, subject, precision_at_k, recall, f1_score, accuracy,
             map_score, ndcg_score, total_chunks, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            metrics.get('subject', 'all'),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('map', 0.0),
            metrics.get('ndcg', 0.0),
            metrics.get('total_chunks', 0),
            metrics.get('notes', ''),
        ))
        conn.commit()
        conn.close()
        return run_id


if __name__ == "__main__":
    wh = DataWarehouse()
    print("Data warehouse module ready (v2 — stable IDs, dedup, metrics table)")