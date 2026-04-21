# ETL - Transform Module (v2 — Robust Noise Removal + Quality Filtering)
# Handles: cleaning, denoising, chunking, metadata enrichment, deduplication

import re
import hashlib
import string
from datetime import datetime
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────────────────────
# NLTK bootstrap
# ─────────────────────────────────────────────────────────────────────────────

for resource, path in [
    ('punkt',         'tokenizers/punkt'),
    ('punkt_tab',     'tokenizers/punkt_tab'),
    ('stopwords',     'corpora/stopwords'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Document Model
# ─────────────────────────────────────────────────────────────────────────────

class Document:
    """Universal document model used throughout the pipeline."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<Document words={len(self.page_content.split())} subject={self.metadata.get('subject','?')}>"


# ─────────────────────────────────────────────────────────────────────────────
# Text Splitter
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTextSplitter:
    """
    Word-based sliding-window text splitter.
    Produces chunks of `chunk_size` words with `chunk_overlap` word overlap.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
            i += step

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result = []
        for doc in documents:
            for chunk_text in self.split_text(doc.page_content):
                result.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                ))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Noise Patterns (compiled once for speed)
# ─────────────────────────────────────────────────────────────────────────────

# Spam / boilerplate phrases (case-insensitive)
_SPAM_PHRASES = [
    # University boilerplate
    r"department\s+of\s+computer\s+science(\s+engineering)?",
    r"srm\s+university(\s+ap)?",
    r"machine\s+learning\s+cse\s+ii",
    r"btechcse",
    r"assist\s*ant\s+professor",
    r"assistant\s+professor",
    r"sivaramakrishna[a-z@.]*",
    r"unit\s+bl[v]?\s+siva\s+rama\s+krishna",
    r"learning\s+objectives",
    # Academic footer/header fragments
    r"all\s+rights\s+reserved",
    r"copyright\s+\d{4}",
    r"prepared\s+by",
    r"presented\s+by",
    r"slide\s+\d+\s+of\s+\d+",
    r"page\s+\d+\s+of\s+\d+",
    r"\[slide\s+\d+\]",
    r"lecture\s+notes?",
]

# Noise patterns (order matters — run before stop-word removal)
_NOISE_PATTERNS = [
    # URLs and emails
    (re.compile(r'https?://\S+', re.IGNORECASE), ' '),
    (re.compile(r'www\.\S+',      re.IGNORECASE), ' '),
    (re.compile(r'\S+@\S+\.\S+',  re.IGNORECASE), ' '),
    # Page / slide numbers    e.g. "Page 3", "Slide 7 of 20"
    (re.compile(r'\bpage\s+\d+\b',         re.IGNORECASE), ' '),
    (re.compile(r'\bslide\s+\d+\b',        re.IGNORECASE), ' '),
    (re.compile(r'\b\d+\s+of\s+\d+\b',    re.IGNORECASE), ' '),
    # LaTeX / math artefacts
    (re.compile(r'\\[a-zA-Z]+\{[^}]*\}'),  ' '),
    (re.compile(r'\$[^$]+\$'),             ' '),
    (re.compile(r'\\[a-zA-Z]+'),           ' '),
    # Non-ASCII garbage (keep basic punctuation/letters/digits)
    (re.compile(r'[^\x00-\x7F]+'),         ' '),
    # Repeated punctuation  e.g. "-----", "======"
    (re.compile(r'[-=_*#|~]{3,}'),        ' '),
    # Standalone single characters on their own (common PDF artefact)
    (re.compile(r'(?<!\w)[^a-zA-Z0-9\s]{2,}(?!\w)'), ' '),
    # Excessive whitespace / newlines
    (re.compile(r'[ \t]{2,}'),  ' '),
    (re.compile(r'\n{3,}'),     '\n\n'),
]

# Compiled spam regex (OR of all phrases)
_SPAM_RE = re.compile(
    '|'.join(_SPAM_PHRASES),
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Domain-Aware Stopwords
# ─────────────────────────────────────────────────────────────────────────────

_CS_KEYWORDS_PRESERVE = {
    # Acronyms / abbreviations used in CS
    'os', 'ai', 'ml', 'it', 'io', 'ip', 'id', 'ui', 'ux', 'db', 'sql',
    'api', 'cpu', 'gpu', 'ram', 'rom', 'lan', 'wan', 'tcp', 'udp', 'dns',
    'http', 'ftp', 'url', 'xml', 'csv', 'json', 'rpc', 'orm', 'erd',
    # Common CS verbs/prepositions that matter
    'as', 'in', 'is', 'at', 'on', 'by', 'or', 'if', 'do', 'no',
    'set', 'get', 'key', 'map', 'log', 'bit', 'bus', 'run', 'mod',
    'not', 'and', 'nor',
}


def _build_stopwords() -> set:
    base = set(stopwords.words('english'))
    # Remove preserved CS keywords from stopwords
    return base - _CS_KEYWORDS_PRESERVE


_STOPWORDS = _build_stopwords()


# ─────────────────────────────────────────────────────────────────────────────
# DataTransformer
# ─────────────────────────────────────────────────────────────────────────────

class DataTransformer:
    """
    Full transformation pipeline:
      1. Advanced noise + spam removal
      2. Conservative number handling (keeps "3NF", "O(n2)", "IPv4")
      3. Domain-aware stopword filtering
      4. Quality filtering (min meaningful words)
      5. Chunking with configurable size/overlap
      6. Metadata enrichment with stable hash-based IDs
      7. Content-level deduplication
    """

    MIN_CHUNK_WORDS = 20          # Discard chunks shorter than this
    MIN_DOCUMENT_CHARS = 50       # Discard documents shorter than this

    def __init__(self):
        self.stats = {
            'documents_in': 0,
            'documents_after_clean': 0,
            'chunks_before_dedup': 0,
            'chunks_after_dedup': 0,
            'chunks_filtered_short': 0,
        }

    # ── Cleaning ─────────────────────────────────────────────────────────────

    def remove_noise(self, text: str) -> str:
        """
        Stage 1 — Remove general PDF/document noise:
        URLs, emails, page numbers, LaTeX, non-ASCII, repeated punctuation.
        """
        for pattern, replacement in _NOISE_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def remove_spam(self, text: str) -> str:
        """Stage 2 — Remove university/document boilerplate phrases."""
        text = _SPAM_RE.sub(' ', text)
        return text

    def normalize_numbers(self, text: str) -> str:
        """
        Stage 3 — Smart number handling.
        KEEP: numbers adjacent to letters ("3NF", "B+", "IPv4", "O(n2)")
        REMOVE: lone standalone integers that are just noise ("1", "23", "456")
        """
        # Remove standalone numbers (not attached to any letter context)
        text = re.sub(r'(?<![a-zA-Z()\[\]])(\b\d{1,4}\b)(?![a-zA-Z%\-\+\*/()\[\]])', ' ', text)
        return text

    def clean_whitespace(self, text: str) -> str:
        """Stage 4 — Normalize whitespace."""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{2,}', '\n', text)
        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """
        Stage 5 — Domain-aware stopword removal.
        Preserves CS acronyms and technically meaningful short words.
        """
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        filtered = []
        for token in tokens:
            lower = token.lower()
            # Keep: not a stopword, OR it's a preserved CS keyword, OR it's alphanumeric
            if lower not in _STOPWORDS or lower in _CS_KEYWORDS_PRESERVE:
                filtered.append(token)

        return ' '.join(filtered)

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline for a single text string."""
        text = self.remove_noise(text)
        text = self.remove_spam(text)
        # Note: Do NOT lowercase or remove punctuation here. 
        # Punctuation is required for sentence splitting in RAG fallback.
        text = self.normalize_numbers(text)
        text = self.clean_whitespace(text)
        return text

    def preprocess_document(self, doc: Document) -> Optional[Document]:
        """
        Clean a document and return None if it's too short after cleaning.
        """
        cleaned = self.clean_text(doc.page_content)
        # We do NOT remove stopwords here to preserve readability for RAG outputs.
        # Stopword weight will be naturally reduced by IDF during retrieval.
        cleaned = self.clean_whitespace(cleaned)

        if len(cleaned) < self.MIN_DOCUMENT_CHARS:
            return None

        return Document(page_content=cleaned, metadata=doc.metadata.copy())

    # ── Chunking ─────────────────────────────────────────────────────────────

    def chunk_documents(self, documents: List[Document],
                        chunk_size: int = 500,
                        chunk_overlap: int = 50) -> List[Document]:
        splitter = SimpleTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        for doc in documents:
            chunks.extend(splitter.split_documents([doc]))
        return chunks

    # ── Metadata ─────────────────────────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        """Stable 12-char hash of content (for dedup + stable chunk IDs)."""
        return hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]

    def add_metadata(self, chunks: List[Document],
                     subject: str = None,
                     topic: str = None,
                     difficulty: str = None) -> List[Document]:
        """Enrich chunks with stable IDs and inferred metadata."""
        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            content_hash = self._content_hash(chunk.page_content)

            # Deduplication
            if content_hash in seen_hashes:
                self.stats['chunks_before_dedup'] += 1
                continue
            seen_hashes.add(content_hash)

            # Quality filter
            word_count = len(chunk.page_content.split())
            if word_count < self.MIN_CHUNK_WORDS:
                self.stats['chunks_filtered_short'] += 1
                continue

            chunk_subject = subject or chunk.metadata.get('subject', 'general')
            chunk.metadata.update({
                # Stable ID: content-hash so re-training is idempotent
                'chunk_id':    f"chunk_{content_hash}",
                'doc_id':      chunk.metadata.get('doc_id', f"doc_{content_hash}"),
                'subject':     chunk_subject,
                'topic':       topic or self.extract_topic(chunk.page_content, chunk_subject),
                'difficulty':  difficulty or self.classify_difficulty(chunk.page_content),
                'word_count':  word_count,
                'content_hash': content_hash,
                'source_file': chunk.metadata.get('source_file', 'unknown'),
                'source_type': chunk.metadata.get('source_type', 'unknown'),
                'processed':   True,
                'timestamp':   datetime.now().isoformat(),
            })
            unique_chunks.append(chunk)

        self.stats['chunks_after_dedup'] = len(unique_chunks)
        return unique_chunks

    # ── Topic / Difficulty Inference ─────────────────────────────────────────

    def extract_topic(self, text: str, subject: str = None) -> str:
        """Keyword-based topic extraction, subject-aware."""
        text_lower = text.lower()

        subject_topics = {
            'DBMS': {
                'indexing':       ['index', 'b-tree', 'b+tree', 'hashing', 'clustered', 'non-clustered'],
                'normalization':  ['normal form', '1nf', '2nf', '3nf', 'bcnf', 'functional dependency', 'decomposition'],
                'transactions':   ['acid', 'commit', 'rollback', 'concurrency', 'serializability', 'isolation'],
                'sql':            ['select', 'insert', 'update', 'delete', 'join', 'where', 'group by', 'having', 'order by'],
                'joins':          ['inner join', 'outer join', 'left join', 'right join', 'natural join', 'cross join'],
                'er_model':       ['entity', 'relationship', 'attribute', 'primary key', 'foreign key', 'cardinality'],
                'relational':     ['relation', 'tuple', 'schema', 'domain', 'degree', 'relational algebra'],
            },
            'OS': {
                'process_scheduling': ['fcfs', 'sjf', 'round robin', 'priority', 'mlq', 'scheduling', 'process state'],
                'memory_management':  ['paging', 'segmentation', 'virtual memory', 'tlb', 'page table', 'frame', 'swapping'],
                'deadlock':           ['deadlock', 'prevention', 'avoidance', 'detection', 'recovery', 'banker'],
                'file_system':        ['inode', 'directory', 'allocation', 'free space', 'fat', 'file descriptor'],
                'synchronization':    ['semaphore', 'mutex', 'monitor', 'critical section', 'race condition'],
                'cpu_architecture':   ['cpu', 'register', 'interrupt', 'cache', 'pipeline'],
            },
            'DataStructures': {
                'arrays':       ['array', 'element', 'index', 'contiguous', 'multidimensional'],
                'stacks':       ['stack', 'push', 'pop', 'lifo', 'postfix', 'infix', 'prefix', 'underflow', 'overflow'],
                'queues':       ['queue', 'enqueue', 'dequeue', 'fifo', 'circular queue', 'deque', 'priority queue'],
                'linked_lists': ['linked list', 'singly', 'doubly', 'circular', 'node', 'pointer', 'traversal'],
                'trees':        ['tree', 'binary tree', 'bst', 'avl', 'heap', 'inorder', 'preorder', 'postorder', 'height'],
                'graphs':       ['graph', 'bfs', 'dfs', 'adjacency', 'vertex', 'edge', 'shortest path', 'spanning tree', 'dijkstra'],
                'sorting':      ['sort', 'bubble', 'merge sort', 'quick sort', 'heap sort', 'insertion sort', 'selection sort', 'radix'],
                'searching':    ['search', 'binary search', 'linear search', 'hashing', 'hash table'],
                'complexity':   ['big o', 'time complexity', 'space complexity', 'asymptotic', 'omega', 'theta', 'worst case', 'best case'],
            },
            'MachineLearning': {
                'supervised':     ['classification', 'regression', 'labeled', 'training data', 'supervised'],
                'unsupervised':   ['clustering', 'kmeans', 'dbscan', 'dimension reduction', 'pca', 'unsupervised'],
                'neural_nets':    ['neural network', 'deep learning', 'backpropagation', 'gradient', 'activation', 'layer', 'weights'],
                'evaluation':     ['accuracy', 'precision', 'recall', 'f1', 'roc', 'auc', 'confusion matrix', 'cross validation'],
                'algorithms':     ['decision tree', 'random forest', 'svm', 'naive bayes', 'knn', 'logistic regression'],
                'preprocessing':  ['feature', 'normalization', 'standardization', 'missing values', 'encoding', 'overfitting'],
            },
        }

        if subject and subject in subject_topics:
            for topic, keywords in subject_topics[subject].items():
                if any(kw in text_lower for kw in keywords):
                    return topic

        # General fallback
        general = {
            'database':        ['database', 'dbms', 'sql', 'table', 'query'],
            'operating_system':['os', 'process', 'thread', 'memory', 'cpu', 'scheduling'],
            'data_structure':  ['array', 'linked list', 'tree', 'graph', 'algorithm'],
            'machine_learning':['machine learning', 'neural', 'training', 'model', 'prediction'],
            'programming':     ['code', 'function', 'class', 'variable', 'loop'],
        }
        for topic, keywords in general.items():
            if any(kw in text_lower for kw in keywords):
                return topic

        return 'general'

    def classify_difficulty(self, text: str) -> str:
        """
        Heuristic difficulty classification based on vocabulary richness
        and presence of advanced technical terms.
        """
        words = text.split()
        word_count = len(words)

        advanced_terms = {
            'serializability', 'normalization', 'concurrency', 'polymorphism',
            'encapsulation', 'deadlock', 'segmentation', 'backpropagation',
            'dimensionality', 'optimization', 'abstraction', 'implementation',
            'complexity', 'asymptotic', 'recurrence', 'dijkstra', 'heuristic',
        }

        # Count advanced terms in content
        advanced_count = sum(1 for w in words if w.lower() in advanced_terms)

        if advanced_count >= 3 or word_count > 300:
            return 'hard'
        elif advanced_count >= 1 or word_count > 100:
            return 'medium'
        else:
            return 'easy'

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def transform_pipeline(self, documents: List[Document],
                           chunk_size: int = 500,
                           chunk_overlap: int = 50) -> List[Document]:
        """
        Complete transformation pipeline:
          1. Clean + denoise each document
          2. Filter out too-short documents
          3. Chunk documents
          4. Add metadata + deduplicate
        """
        self.stats['documents_in'] = len(documents)
        print(f"  [Transform] Input: {len(documents)} documents")

        # Step 1: Clean
        processed_docs = []
        for doc in documents:
            cleaned = self.preprocess_document(doc)
            if cleaned is not None:
                processed_docs.append(cleaned)

        self.stats['documents_after_clean'] = len(processed_docs)
        skipped = len(documents) - len(processed_docs)
        print(f"  [Transform] After cleaning: {len(processed_docs)} docs ({skipped} too short / empty, discarded)")

        # Step 2: Chunk
        chunks = self.chunk_documents(processed_docs, chunk_size, chunk_overlap)
        print(f"  [Transform] Chunks created: {len(chunks)}")

        # Step 3: Metadata + dedup
        enriched = self.add_metadata(chunks)
        deduped = self.stats['chunks_before_dedup']
        filtered = self.stats['chunks_filtered_short']
        print(f"  [Transform] After dedup: {len(enriched)} chunks "
              f"({deduped} duplicates removed, {filtered} too-short filtered)")

        return enriched

    def print_stats(self):
        s = self.stats
        print("\n  ─── Transform Stats ───")
        print(f"  Documents in          : {s['documents_in']}")
        print(f"  After cleaning        : {s['documents_after_clean']}")
        print(f"  Duplicate chunks      : {s['chunks_before_dedup']}")
        print(f"  Short chunks filtered : {s['chunks_filtered_short']}")
        print(f"  Final chunks          : {s['chunks_after_dedup']}")


if __name__ == "__main__":
    transformer = DataTransformer()
    print("Data transformation module ready (v2 — noise removal + dedup)")