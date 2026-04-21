# Analytics - Evaluator Module (NEW)
# Full ML evaluation suite: Classification, Retrieval, and Quality metrics
#
# Provides:
#   - Precision, Recall, F1, Accuracy
#   - Confusion Matrix (per-class)
#   - MAP, MRR, NDCG
#   - Answer Quality scoring
#   - Per-subject evaluation report
#   - JSON + console report output

import os
import json
import math
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from etl.load import DataWarehouse
from rag.retriever import DocumentRetriever


# ─────────────────────────────────────────────────────────────────────────────
# Console colours
# ─────────────────────────────────────────────────────────────────────────────

GREEN   = '\033[92m'
BLUE    = '\033[94m'
YELLOW  = '\033[93m'
RED     = '\033[91m'
CYAN    = '\033[96m'
MAGENTA = '\033[95m'
RESET   = '\033[0m'
BOLD    = '\033[1m'
DIM     = '\033[2m'


# ─────────────────────────────────────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationMetrics:
    """
    Compute standard binary/multi-class classification metrics from
    a list of (predicted_label, true_label) tuples.
    """

    def __init__(self, predictions: List[str], ground_truths: List[str]):
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must be the same length")
        self.predictions   = predictions
        self.ground_truths = ground_truths
        self.labels        = sorted(set(ground_truths) | set(predictions))

    # ── Binary helpers ────────────────────────────────────────────────────────

    def _binary_counts(self, pos_label: str) -> Tuple[int, int, int, int]:
        """Return (TP, FP, FN, TN) for a single label treated as positive."""
        TP = FP = FN = TN = 0
        for pred, true in zip(self.predictions, self.ground_truths):
            is_pred_pos = (pred  == pos_label)
            is_true_pos = (true  == pos_label)
            if is_pred_pos and is_true_pos:  TP += 1
            elif is_pred_pos:                FP += 1
            elif is_true_pos:                FN += 1
            else:                            TN += 1
        return TP, FP, FN, TN

    # ── Per-class metrics ─────────────────────────────────────────────────────

    def per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for label in self.labels:
            TP, FP, FN, TN = self._binary_counts(label)
            precision = TP / (TP + FP) if (TP + FP) else 0.0
            recall    = TP / (TP + FN) if (TP + FN) else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) else 0.0)
            accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0
            results[label] = {
                'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
                'precision': round(precision, 4),
                'recall':    round(recall,    4),
                'f1':        round(f1,        4),
                'accuracy':  round(accuracy,  4),
            }
        return results

    # ── Macro averages ────────────────────────────────────────────────────────

    def macro_precision(self) -> float:
        pcm = self.per_class_metrics()
        return round(sum(v['precision'] for v in pcm.values()) / len(pcm), 4) if pcm else 0.0

    def macro_recall(self) -> float:
        pcm = self.per_class_metrics()
        return round(sum(v['recall'] for v in pcm.values()) / len(pcm), 4) if pcm else 0.0

    def macro_f1(self) -> float:
        pcm = self.per_class_metrics()
        return round(sum(v['f1'] for v in pcm.values()) / len(pcm), 4) if pcm else 0.0

    def overall_accuracy(self) -> float:
        correct = sum(1 for p, t in zip(self.predictions, self.ground_truths) if p == t)
        return round(correct / len(self.predictions), 4) if self.predictions else 0.0

    # ── Confusion matrix ──────────────────────────────────────────────────────

    def confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        confusion_matrix[true_label][pred_label] = count
        """
        matrix = {lab: {lab2: 0 for lab2 in self.labels} for lab in self.labels}
        for pred, true in zip(self.predictions, self.ground_truths):
            if true in matrix and pred in matrix[true]:
                matrix[true][pred] += 1
        return matrix

    def summary(self) -> Dict:
        return {
            'overall_accuracy':  self.overall_accuracy(),
            'macro_precision':   self.macro_precision(),
            'macro_recall':      self.macro_recall(),
            'macro_f1':          self.macro_f1(),
            'per_class':         self.per_class_metrics(),
            'confusion_matrix':  self.confusion_matrix(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Metrics
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalMetrics:
    """
    Compute IR metrics for a single query-result pair.
    All methods operate on sets of relevant document IDs / strings.
    """

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        top_k = retrieved[:k]
        rel_set = set(r.lower() for r in relevant)
        hits = sum(1 for doc in top_k if doc.lower() in rel_set or
                   any(r in doc.lower() for r in rel_set))
        return round(hits / k, 4) if k else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        top_k = retrieved[:k]
        rel_set = set(r.lower() for r in relevant)
        hits = sum(1 for doc in top_k if doc.lower() in rel_set or
                   any(r in doc.lower() for r in rel_set))
        return round(hits / len(relevant), 4) if relevant else 0.0

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        p = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        r = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        rel_set = set(r.lower() for r in relevant)
        ap = 0.0
        hits = 0
        for rank, doc in enumerate(retrieved, start=1):
            matched = doc.lower() in rel_set or any(r in doc.lower() for r in rel_set)
            if matched:
                hits += 1
                ap += hits / rank
        return round(ap / len(relevant), 4) if relevant else 0.0

    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        rel_set = set(r.lower() for r in relevant)
        for rank, doc in enumerate(retrieved, start=1):
            if doc.lower() in rel_set or any(r in doc.lower() for r in rel_set):
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        rel_set = set(r.lower() for r in relevant)
        dcg  = 0.0
        idcg = 0.0
        n_rel = min(len(relevant), k)

        for rank in range(1, k + 1):
            # IDCG: ideal order
            if rank <= n_rel:
                idcg += 1.0 / math.log2(rank + 1)
            # DCG: actual retrieval
            if rank <= len(retrieved):
                doc = retrieved[rank - 1]
                if doc.lower() in rel_set or any(r in doc.lower() for r in rel_set):
                    dcg += 1.0 / math.log2(rank + 1)

        return round(dcg / idcg, 4) if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Answer Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

class QualityMetrics:
    """Heuristic-based answer quality scoring."""

    @staticmethod
    def keyword_overlap(answer: str, context_chunks: List[dict]) -> float:
        """
        Measure how many keywords from context appear in the answer.
        Score ∈ [0, 1].
        """
        if not context_chunks or not answer:
            return 0.0

        context_words = set()
        for chunk in context_chunks:
            content = chunk.get('content', '')
            words   = [w.lower() for w in content.split() if len(w) > 4]
            context_words.update(words)

        answer_words = set(w.lower() for w in answer.split() if len(w) > 4)
        if not context_words:
            return 0.0

        overlap = context_words & answer_words
        return round(len(overlap) / len(context_words), 4)

    @staticmethod
    def completeness_score(answer: str, query: str) -> float:
        """
        Check if the answer addresses query keywords.
        Score ∈ [0, 1].
        """
        if not answer or not query:
            return 0.0

        query_words  = set(w.lower() for w in query.split() if len(w) > 3)
        answer_words = set(w.lower() for w in answer.split())

        addressed = query_words & answer_words
        return round(len(addressed) / len(query_words), 4) if query_words else 0.0

    @staticmethod
    def composite_quality_score(answer: str,
                                query: str,
                                context_chunks: List[dict]) -> Dict[str, float]:
        """Compute all quality metrics and return a composite score."""
        overlap      = QualityMetrics.keyword_overlap(answer, context_chunks)
        completeness = QualityMetrics.completeness_score(answer, query)
        word_count   = len(answer.split())

        has_examples = float('example' in answer.lower() or
                             'for example' in answer.lower() or
                             'e.g.' in answer.lower())
        has_structure = float(any(c in answer for c in ['1.', '2.', '\n-', '\n*', 'First', 'Second']))
        length_score  = min(1.0, max(0.0, (word_count - 20) / 200))

        # Weighted composite
        composite = (
            0.35 * overlap +
            0.25 * completeness +
            0.15 * has_examples +
            0.10 * has_structure +
            0.15 * length_score
        )

        return {
            'keyword_overlap':   round(overlap, 4),
            'completeness':      round(completeness, 4),
            'has_examples':      has_examples,
            'has_structure':     has_structure,
            'length_score':      round(length_score, 4),
            'composite_score':   round(composite, 4),
            'answer_word_count': word_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Master Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Orchestrates all evaluation:
      1. Data-coverage analysis (chunk stats per subject/difficulty/topic)
      2. Retrieval metrics  (Precision, Recall, F1, MAP, NDCG)
         Uses built-in test queries with known relevant subjects
      3. Classification metrics (topic assignment accuracy — simulation)
      4. Quality metrics (answer completeness / keyword overlap)
      5. Full report: console + JSON
    """

    # Built-in test suite: (query, relevant_subject, sample_relevant_keywords)
    TEST_QUERIES: List[Tuple[str, str, List[str]]] = [
        ("What is normalization in databases?",         "DBMS",           ["normalization", "normal", "dependency"]),
        ("Explain ACID properties in transactions",     "DBMS",           ["acid", "commit", "rollback", "transaction"]),
        ("What is process scheduling in OS?",           "OS",             ["scheduling", "process", "fcfs", "cpu"]),
        ("Explain virtual memory and paging",           "OS",             ["virtual", "paging", "page", "memory"]),
        ("What is a binary search tree?",               "DataStructures", ["binary", "tree", "search", "bst"]),
        ("Explain BFS and DFS in graphs",               "DataStructures", ["graph", "bfs", "dfs", "vertex"]),
        ("What is a linked list?",                      "DataStructures", ["linked", "list", "node", "pointer"]),
        ("What is machine learning classification?",    "MachineLearning",["classification", "label", "supervised"]),
    ]

    def __init__(self,
                 warehouse: DataWarehouse = None,
                 retriever: DocumentRetriever = None):
        self.warehouse = warehouse or DataWarehouse()
        self.warehouse.get_vectorstore()
        self.retriever = retriever or DocumentRetriever(self.warehouse)
        self.report: Dict = {}

    # ── Data Coverage ─────────────────────────────────────────────────────────

    def analyze_data_coverage(self) -> Dict:
        """Analyze the content stored in the warehouse."""
        try:
            chunks = self.warehouse.query_sql("SELECT subject, topic, difficulty, word_count FROM chunks")
        except Exception as e:
            return {'error': str(e)}

        if not chunks:
            return {'error': 'No data in warehouse'}

        subjects    = defaultdict(int)
        topics      = defaultdict(int)
        difficulties = defaultdict(int)
        total_words = 0

        for subject, topic, difficulty, word_count in chunks:
            subjects[subject or 'unknown'] += 1
            topics[topic or 'unknown']     += 1
            difficulties[difficulty or 'unknown'] += 1
            total_words += (word_count or 0)

        return {
            'total_chunks':       len(chunks),
            'total_words':        total_words,
            'avg_words_per_chunk': round(total_words / len(chunks), 1) if chunks else 0,
            'by_subject':         dict(subjects),
            'by_topic':           dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:20]),
            'by_difficulty':      dict(difficulties),
        }

    # ── Retrieval Evaluation ──────────────────────────────────────────────────

    def evaluate_retrieval(self, k: int = 5) -> Dict:
        """Run all built-in test queries and compute aggregate retrieval metrics."""
        query_results = []
        rm = RetrievalMetrics()

        for query, expected_subject, relevant_keywords in self.TEST_QUERIES:
            retrieved_docs_raw = self.retriever.retrieve_similar(query, k=k)

            retrieved_contents = [doc.page_content for doc, _ in retrieved_docs_raw]
            retrieved_subjects = [doc.metadata.get('subject', '')
                                  for doc, _ in retrieved_docs_raw]

            p_k   = rm.precision_at_k(retrieved_contents, relevant_keywords, k)
            r_k   = rm.recall_at_k(retrieved_contents, relevant_keywords, k)
            f1_k  = rm.f1_at_k(retrieved_contents, relevant_keywords, k)
            ap    = rm.average_precision(retrieved_contents, relevant_keywords)
            rr    = rm.reciprocal_rank(retrieved_contents, relevant_keywords)
            ndcg  = rm.ndcg_at_k(retrieved_contents, relevant_keywords, k)

            # Subject accuracy: was expected subject in retrieved docs?
            subj_hit = int(expected_subject in retrieved_subjects)

            query_results.append({
                'query':            query,
                'expected_subject': expected_subject,
                'precision':        p_k,
                'recall':           r_k,
                'f1':               f1_k,
                'average_precision': ap,
                'reciprocal_rank':  rr,
                'ndcg':             ndcg,
                'subject_hit':      subj_hit,
                'retrieved_count':  len(retrieved_docs_raw),
            })

        def _avg(field):
            return round(sum(r[field] for r in query_results) / len(query_results), 4) \
                   if query_results else 0.0

        return {
            'k':                  k,
            'total_queries':      len(query_results),
            'macro_precision':    _avg('precision'),
            'macro_recall':       _avg('recall'),
            'macro_f1':           _avg('f1'),
            'MAP':                _avg('average_precision'),
            'MRR':                _avg('reciprocal_rank'),
            'mean_ndcg':          _avg('ndcg'),
            'subject_accuracy':   _avg('subject_hit'),
            'per_query':          query_results,
        }

    # ── Classification Metrics (Topic Assignment) ─────────────────────────────

    def evaluate_topic_classification(self) -> Dict:
        """
        Treat topic assignment as a classification task.
        Uses chunks already stored in the warehouse.
        Compares assigned topic vs. the most-likely topic re-inferred from content.
        """
        # Import transformer for re-inference
        from etl.transform import DataTransformer
        transformer = DataTransformer()

        try:
            rows = self.warehouse.query_sql(
                "SELECT subject, topic, content FROM chunks LIMIT 200"
            )
        except Exception as e:
            return {'error': str(e)}

        if not rows:
            return {'error': 'No chunks for classification evaluation'}

        predictions  = []
        ground_truths = []

        for subject, stored_topic, content in rows:
            if not content:
                continue
            inferred_topic = transformer.extract_topic(content[:500], subject)
            stored_clean   = (stored_topic or 'general').strip()

            predictions.append(inferred_topic)
            ground_truths.append(stored_clean)

        if not predictions:
            return {'error': 'No valid samples'}

        cm = ClassificationMetrics(predictions, ground_truths)
        return {
            'samples_evaluated': len(predictions),
            **cm.summary(),
        }

    # ── Noise / Quality Analysis ──────────────────────────────────────────────

    def analyze_chunk_quality(self) -> Dict:
        """Check chunks for remaining noise indicators."""
        try:
            rows = self.warehouse.query_sql("SELECT content FROM chunks LIMIT 500")
        except Exception as e:
            return {'error': str(e)}

        if not rows:
            return {'info': 'No chunks found'}

        import re
        noise_patterns = {
            'urls':            re.compile(r'https?://', re.I),
            'emails':          re.compile(r'\S+@\S+\.\S+', re.I),
            'page_numbers':    re.compile(r'\bpage\s+\d+\b', re.I),
            'latex':           re.compile(r'\\[a-zA-Z]+'),
            'non_ascii':       re.compile(r'[^\x00-\x7F]'),
            'repeated_punct':  re.compile(r'[-=_*]{4,}'),
        }

        total  = len(rows)
        flagged_counts = {k: 0 for k in noise_patterns}

        for (content,) in rows:
            if not content:
                continue
            for pattern_name, pattern in noise_patterns.items():
                if pattern.search(content):
                    flagged_counts[pattern_name] += 1

        return {
            'total_chunks_checked': total,
            'noise_flags': {
                k: {'count': v, 'pct': round(v / total * 100, 1)}
                for k, v in flagged_counts.items()
            },
            'clean_pct': round(
                100 * (1 - sum(flagged_counts.values()) / (total * len(noise_patterns))),
                1
            ),
        }

    # ── Full Report ───────────────────────────────────────────────────────────

    def generate_evaluation_report(self, k: int = 5, save_path: str = "data/eval_report.json") -> Dict:
        """Run all evaluations and produce a full report."""
        start = time.time()
        print(f"\n{CYAN}{BOLD}{'═'*60}{RESET}")
        print(f"{CYAN}{BOLD}  📊  EVALUATION REPORT{RESET}")
        print(f"{CYAN}{BOLD}{'═'*60}{RESET}\n")

        # 1. Coverage
        print(f"{BLUE}▶ Analyzing data coverage...{RESET}")
        coverage = self.analyze_data_coverage()
        self._print_coverage(coverage)

        # 2. Retrieval
        print(f"{BLUE}▶ Running retrieval evaluation (k={k})...{RESET}")
        retrieval = self.evaluate_retrieval(k=k)
        self._print_retrieval(retrieval)

        # 3. Classification
        print(f"{BLUE}▶ Evaluating topic classification...{RESET}")
        classification = self.evaluate_topic_classification()
        self._print_classification(classification)

        # 4. Noise
        print(f"{BLUE}▶ Checking chunk quality / noise signatures...{RESET}")
        noise = self.analyze_chunk_quality()
        self._print_noise(noise)

        elapsed = round(time.time() - start, 2)
        self.report = {
            'generated_at':    datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'data_coverage':   coverage,
            'retrieval':       retrieval,
            'classification':  classification,
            'noise_analysis':  noise,
        }

        # Save JSON
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, default=str)

        # Save to DB
        if 'macro_f1' in retrieval:
            self.warehouse.save_evaluation_metrics({
                'subject':        'all',
                'precision':      retrieval['macro_precision'],
                'recall':         retrieval['macro_recall'],
                'f1':             retrieval['macro_f1'],
                'accuracy':       classification.get('overall_accuracy', 0.0),
                'map':            retrieval['MAP'],
                'ndcg':           retrieval['mean_ndcg'],
                'total_chunks':   coverage.get('total_chunks', 0),
                'notes':          f"Auto eval — k={k}",
            })

        print(f"\n{GREEN}{BOLD}  ✅ Report saved → {save_path}{RESET}")
        print(f"  Evaluation completed in {elapsed}s\n")
        return self.report

    # ── Print Helpers ─────────────────────────────────────────────────────────

    def _print_coverage(self, c: Dict):
        if 'error' in c:
            print(f"  {RED}Error: {c['error']}{RESET}")
            return
        print(f"  Total chunks    : {BOLD}{c.get('total_chunks', 0)}{RESET}")
        print(f"  Total words     : {c.get('total_words', 0):,}")
        print(f"  Avg words/chunk : {c.get('avg_words_per_chunk', 0)}")
        print(f"\n  By Subject:")
        for subj, cnt in c.get('by_subject', {}).items():
            bar = '█' * min(cnt // 5 + 1, 20)
            print(f"    {subj:20s} {cnt:5d}  {DIM}{bar}{RESET}")
        print(f"\n  By Difficulty:")
        for diff, cnt in c.get('by_difficulty', {}).items():
            print(f"    {diff:10s} {cnt:5d}")
        print()

    def _print_retrieval(self, r: Dict):
        if 'error' in r:
            print(f"  {RED}Error: {r['error']}{RESET}")
            return
        print(f"\n  {BOLD}Retrieval Metrics (k={r['k']}, {r['total_queries']} queries){RESET}")
        print(f"  {'─'*42}")

        metrics = [
            ("Precision",        r.get('macro_precision', 0)),
            ("Recall",           r.get('macro_recall',    0)),
            ("F1 Score",         r.get('macro_f1',        0)),
            ("MAP",              r.get('MAP',             0)),
            ("MRR",              r.get('MRR',             0)),
            ("Mean NDCG",        r.get('mean_ndcg',       0)),
            ("Subject Accuracy", r.get('subject_accuracy',0)),
        ]

        for name, value in metrics:
            bar_len = int(value * 20)
            bar_color = GREEN if value >= 0.6 else (YELLOW if value >= 0.3 else RED)
            bar = bar_color + '█' * bar_len + DIM + '░' * (20 - bar_len) + RESET
            print(f"  {name:18s}  {bar}  {BOLD}{value:.4f}{RESET}")

        print(f"\n  {DIM}Per-Query Breakdown:{RESET}")
        for qr in r.get('per_query', []):
            status = GREEN + '✓' + RESET if qr['f1'] > 0 else RED + '✗' + RESET
            print(f"    {status} [{qr['expected_subject']:15s}] F1={qr['f1']:.3f}  "
                  f"P={qr['precision']:.3f}  R={qr['recall']:.3f}  "
                  f"NDCG={qr['ndcg']:.3f}  | {qr['query'][:50]}")
        print()

    def _print_classification(self, c: Dict):
        if 'error' in c:
            print(f"  {RED}Error: {c['error']}{RESET}")
            return
        print(f"\n  {BOLD}Topic Classification (n={c.get('samples_evaluated', 0)} samples){RESET}")
        print(f"  {'─'*42}")
        print(f"  Overall Accuracy  : {GREEN}{BOLD}{c.get('overall_accuracy', 0):.4f}{RESET}")
        print(f"  Macro Precision   : {c.get('macro_precision', 0):.4f}")
        print(f"  Macro Recall      : {c.get('macro_recall',    0):.4f}")
        print(f"  Macro F1          : {c.get('macro_f1',        0):.4f}")

        pc = c.get('per_class', {})
        if pc:
            print(f"\n  {DIM}Top per-class F1:{RESET}")
            sorted_pc = sorted(pc.items(), key=lambda x: x[1]['f1'], reverse=True)[:10]
            for label, m in sorted_pc:
                f1_color = GREEN if m['f1'] >= 0.5 else (YELLOW if m['f1'] >= 0.2 else RED)
                print(f"    {label:25s}  "
                      f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
                      f"F1={f1_color}{m['f1']:.3f}{RESET}  "
                      f"(TP={m['TP']} FP={m['FP']} FN={m['FN']})")
        print()

    def _print_noise(self, n: Dict):
        if 'error' in n or 'info' in n:
            msg = n.get('error') or n.get('info')
            print(f"  {YELLOW}{msg}{RESET}\n")
            return
        print(f"\n  {BOLD}Chunk Quality / Noise Check ({n.get('total_chunks_checked', 0)} samples){RESET}")
        print(f"  {'─'*42}")
        print(f"  Clean estimate : {GREEN}{BOLD}{n.get('clean_pct', 0)}%{RESET}")
        for noise_type, info in n.get('noise_flags', {}).items():
            count = info['count']
            pct   = info['pct']
            color = GREEN if count == 0 else (YELLOW if pct < 5 else RED)
            print(f"  {noise_type:20s} : {color}{count:4d} chunks ({pct}%){RESET}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    # Ensure project root is on sys.path when running as a script
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    # Re-import after path fix (needed when run as __main__)
    from etl.load import DataWarehouse
    from rag.retriever import DocumentRetriever

    parser = argparse.ArgumentParser(description="AI Teaching Assistant — Evaluation Suite")
    parser.add_argument('--k', type=int, default=5, help='Top-k for retrieval evaluation')
    parser.add_argument('--output', type=str, default='data/eval_report.json',
                        help='Path to save JSON evaluation report')
    args = parser.parse_args()

    evaluator = Evaluator()
    evaluator.generate_evaluation_report(k=args.k, save_path=args.output)
