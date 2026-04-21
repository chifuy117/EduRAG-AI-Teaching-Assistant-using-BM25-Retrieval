# RAG - Retriever Module (v2 — Full Metrics: F1, Precision, Recall, Accuracy, MAP, NDCG)

import math
from typing import List, Tuple, Dict, Optional
from etl.load import DataWarehouse


# ─────────────────────────────────────────────────────────────────────────────
# Document Retriever
# ─────────────────────────────────────────────────────────────────────────────

class DocumentRetriever:
    """
    Keyword-based document retriever backed by DataWarehouse.
    Provides both retrieval methods and a full evaluation suite.
    """

    def __init__(self, warehouse: DataWarehouse = None):
        self.warehouse = warehouse or DataWarehouse()
        self.warehouse.get_vectorstore()   # Load existing vector store from disk

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_similar(self, query: str,
                         k: int = 3,
                         score_threshold: float = 0.0) -> List[Tuple]:
        """Return k most similar (document, score) tuples."""
        if not self.warehouse.chunks_data:
            print("  [Retriever] Vector store is empty — run training first.")
            return []

        results = self.warehouse.search_similar(query, k=k)
        return [(doc, score) for doc, score in results if score >= score_threshold]

    def retrieve_by_metadata(self, query: str,
                             subject: str = None,
                             topic: str = None,
                             difficulty: str = None,
                             k: int = 5) -> List[Tuple]:
        """Retrieve documents filtered by metadata. Falls back to unfiltered if not enough results."""
        # Get extra candidates (wide net)
        candidates = self.retrieve_similar(query, k=max(k * 6, 30))

        # Subject-filtered pass
        filtered = []
        for doc, score in candidates:
            meta = doc.metadata
            if subject    and meta.get('subject')    != subject:    continue
            if topic      and meta.get('topic')      != topic:      continue
            if difficulty and meta.get('difficulty') != difficulty: continue
            filtered.append((doc, score))
            if len(filtered) >= k:
                break

        # If not enough results, relax subject filter and pull from all candidates
        if len(filtered) < max(1, k // 2):
            seen_ids = {id(doc) for doc, _ in filtered}
            for doc, score in candidates:
                if id(doc) in seen_ids:
                    continue
                if topic      and doc.metadata.get('topic')      != topic:      continue
                if difficulty and doc.metadata.get('difficulty') != difficulty: continue
                filtered.append((doc, score))
                if len(filtered) >= k:
                    break

        # Last resort: return top-k unfiltered
        if not filtered:
            filtered = candidates[:k]

        return filtered[:k]

    def get_relevant_chunks(self, query: str,
                            filters: dict = None) -> List[dict]:
        """Return relevant chunks as plain dicts (for RAG pipeline)."""
        k = (filters or {}).get('k', 5)
        if filters:
            results = self.retrieve_by_metadata(
                query=query,
                subject=filters.get('subject'),
                topic=filters.get('topic'),
                difficulty=filters.get('difficulty'),
                k=k,
            )
        else:
            results = self.retrieve_similar(query, k=k)

        return [
            {
                'content':         doc.page_content,
                'metadata':        doc.metadata,
                'relevance_score': float(score),
                'source_file':     doc.metadata.get('source_file', 'unknown'),
                'source':          doc.metadata.get('source_file',
                                       doc.metadata.get('source', 'unknown')),
            }
            for doc, score in results
        ]

    # ── Evaluation Suite ──────────────────────────────────────────────────────

    @staticmethod
    def _is_relevant(content: str, ground_truth_chunks: List[str]) -> bool:
        """Check if retrieved content matches any ground-truth chunk."""
        content_lower = content.lower()
        return any(gt.lower() in content_lower for gt in ground_truth_chunks)

    def evaluate_retrieval(self, query: str,
                           ground_truth_chunks: List[str],
                           k: int = 5) -> Dict[str, float]:
        """
        Full retrieval quality evaluation.

        Returns:
            precision      — TP / (TP + FP)
            recall         — TP / (TP + FN)
            f1             — 2 * P * R / (P + R)
            accuracy       — (TP + TN) / (TP + TN + FP + FN)
            precision_at_k — precision restricted to top-k results
            average_precision — area under P-R curve (for MAP)
            ndcg           — Normalised Discounted Cumulative Gain
        """
        retrieved = self.retrieve_similar(query, k=k)
        retrieved_docs = [doc for doc, _ in retrieved]
        n_retrieved = len(retrieved_docs)

        # ── Basic counts ──────────────────────────────────────────────────────
        TP = sum(1 for doc in retrieved_docs
                 if self._is_relevant(doc.page_content, ground_truth_chunks))
        FP = n_retrieved - TP
        FN = max(0, len(ground_truth_chunks) - TP)

        # TN: chunks NOT retrieved AND NOT in ground truth
        total_chunks = len(self.warehouse.chunks_data)
        TN = max(0, total_chunks - n_retrieved - len(ground_truth_chunks))

        # ── Precision / Recall / F1 ───────────────────────────────────────────
        precision    = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall       = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1           = (2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0)
        accuracy     = ((TP + TN) / (TP + TN + FP + FN)
                        if (TP + TN + FP + FN) > 0 else 0.0)
        precision_at_k = precision  # same when evaluating exactly k results

        # ── Average Precision (for MAP) ───────────────────────────────────────
        ap_sum = 0.0
        rel_count = 0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if self._is_relevant(doc.page_content, ground_truth_chunks):
                rel_count += 1
                ap_sum += rel_count / rank
        average_precision = (ap_sum / len(ground_truth_chunks)
                             if ground_truth_chunks else 0.0)

        # ── NDCG ─────────────────────────────────────────────────────────────
        dcg  = sum(
            (1.0 / math.log2(rank + 1))
            for rank, doc in enumerate(retrieved_docs, start=1)
            if self._is_relevant(doc.page_content, ground_truth_chunks)
        )
        n_relevant = min(len(ground_truth_chunks), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_relevant + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            'precision':        round(precision,         4),
            'recall':           round(recall,            4),
            'f1':               round(f1,                4),
            'accuracy':         round(accuracy,          4),
            'precision_at_k':  round(precision_at_k,    4),
            'average_precision':round(average_precision, 4),
            'ndcg':             round(ndcg,              4),
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'retrieved_count':  n_retrieved,
        }

    def evaluate_batch(self, query_ground_truth_pairs: List[Tuple[str, List[str]]],
                       k: int = 5) -> Dict:
        """
        Evaluate over multiple queries and report:
          - Per-query metrics
          - Macro-averaged Precision / Recall / F1 / Accuracy
          - MAP (Mean Average Precision)
          - Mean NDCG
        """
        per_query = []
        for query, gt_chunks in query_ground_truth_pairs:
            metrics = self.evaluate_retrieval(query, gt_chunks, k=k)
            metrics['query'] = query
            per_query.append(metrics)

        if not per_query:
            return {'error': 'No queries evaluated'}

        def _avg(field):
            return round(sum(m[field] for m in per_query) / len(per_query), 4)

        return {
            'num_queries':  len(per_query),
            'k':            k,
            'macro_precision':  _avg('precision'),
            'macro_recall':     _avg('recall'),
            'macro_f1':         _avg('f1'),
            'macro_accuracy':   _avg('accuracy'),
            'MAP':              _avg('average_precision'),
            'mean_ndcg':        _avg('ndcg'),
            'per_query':        per_query,
        }

    def print_metrics(self, metrics: Dict):
        """Pretty-print evaluation metrics."""
        print("\n  ╔══════════════════════════════════════╗")
        print("  ║        RETRIEVAL METRICS REPORT      ║")
        print("  ╚══════════════════════════════════════╝")

        if 'macro_precision' in metrics:
            # Batch report
            print(f"  Queries evaluated : {metrics['num_queries']} (k={metrics['k']})")
            print(f"  ─────────────────────────────────────")
            print(f"  Macro Precision   : {metrics['macro_precision']:.4f}")
            print(f"  Macro Recall      : {metrics['macro_recall']:.4f}")
            print(f"  Macro F1 Score    : {metrics['macro_f1']:.4f}")
            print(f"  Macro Accuracy    : {metrics['macro_accuracy']:.4f}")
            print(f"  MAP               : {metrics['MAP']:.4f}")
            print(f"  Mean NDCG         : {metrics['mean_ndcg']:.4f}")
        else:
            # Single-query report
            print(f"  Precision         : {metrics['precision']:.4f}")
            print(f"  Recall            : {metrics['recall']:.4f}")
            print(f"  F1 Score          : {metrics['f1']:.4f}")
            print(f"  Accuracy          : {metrics['accuracy']:.4f}")
            print(f"  Precision@K       : {metrics['precision_at_k']:.4f}")
            print(f"  Avg Precision     : {metrics['average_precision']:.4f}")
            print(f"  NDCG              : {metrics['ndcg']:.4f}")
            print(f"  ─── Confusion Matrix ──────────────────")
            print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")

        print()


if __name__ == "__main__":
    retriever = DocumentRetriever()
    print("Document retriever module ready (v2 — full metrics)")