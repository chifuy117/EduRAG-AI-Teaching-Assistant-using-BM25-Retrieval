# Experiments Module (v2 — Fixed imports, real F1/Precision/Recall/Accuracy metrics)
# Evaluates different ETL and RAG configurations and reports rigorous metrics

import json
from typing import Dict, List   # ← FIXED: was missing, caused NameError
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataWarehouse
from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator
from analytics.evaluator import Evaluator, RetrievalMetrics, QualityMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Test Queries with ground-truth keywords (for real metric computation)
# ─────────────────────────────────────────────────────────────────────────────

TEST_SUITE: List[Dict] = [
    {
        'query':    "What is normalization in databases?",
        'subject':  "DBMS",
        'relevant': ["normalization", "normal form", "dependency", "1nf", "2nf"],
    },
    {
        'query':    "Explain CPU scheduling algorithms",
        'subject':  "OS",
        'relevant': ["scheduling", "fcfs", "sjf", "round robin", "priority"],
    },
    {
        'query':    "What is a binary search tree?",
        'subject':  "DataStructures",
        'relevant': ["binary", "tree", "search", "bst", "node"],
    },
    {
        'query':    "Explain ACID properties",
        'subject':  "DBMS",
        'relevant': ["acid", "atomicity", "consistency", "isolation", "durability"],
    },
    {
        'query':    "What is virtual memory?",
        'subject':  "OS",
        'relevant': ["virtual", "memory", "paging", "page", "frame"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRunner
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    def __init__(self):
        self.results: List[Dict] = []
        self.rm = RetrievalMetrics()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _evaluate_retrieval_metrics(self, retriever: DocumentRetriever,
                                    k: int = 3) -> Dict:
        """
        Run all TEST_SUITE queries through `retriever` and compute
        Precision@K, Recall@K, F1@K, MAP, MRR, NDCG.
        """
        per_query = []
        for case in TEST_SUITE:
            raw = retriever.retrieve_similar(case['query'], k=k)
            retrieved_contents = [doc.page_content for doc, _ in raw]

            p   = self.rm.precision_at_k(retrieved_contents, case['relevant'], k)
            r   = self.rm.recall_at_k(retrieved_contents,   case['relevant'], k)
            f1  = self.rm.f1_at_k(retrieved_contents,       case['relevant'], k)
            ap  = self.rm.average_precision(retrieved_contents, case['relevant'])
            rr  = self.rm.reciprocal_rank(retrieved_contents,   case['relevant'])
            ndcg= self.rm.ndcg_at_k(retrieved_contents, case['relevant'], k)

            per_query.append({
                'query':             case['query'],
                'precision':         p,
                'recall':            r,
                'f1':                f1,
                'average_precision': ap,
                'reciprocal_rank':   rr,
                'ndcg':              ndcg,
            })

        def _avg(field):
            return round(sum(q[field] for q in per_query) / len(per_query), 4)

        return {
            'macro_precision': _avg('precision'),
            'macro_recall':    _avg('recall'),
            'macro_f1':        _avg('f1'),
            'MAP':             _avg('average_precision'),
            'MRR':             _avg('reciprocal_rank'),
            'mean_ndcg':       _avg('ndcg'),
            'per_query':       per_query,
        }

    # ── Experiment 1: Chunk Size ──────────────────────────────────────────────

    def run_chunk_size_experiment(self, documents, chunk_sizes: List[int] = None) -> List[Dict]:
        """Compare retrieval quality across multiple chunk sizes."""
        if chunk_sizes is None:
            chunk_sizes = [200, 500, 1000]

        print("\n🧪  Experiment: Chunk Size Comparison")
        print("   Chunk sizes:", chunk_sizes)

        generator  = AnswerGenerator()
        results    = []

        for chunk_size in chunk_sizes:
            print(f"\n  Testing chunk_size={chunk_size}…")

            transformer = DataTransformer()
            chunks      = transformer.transform_pipeline(documents, chunk_size=chunk_size)

            warehouse   = DataWarehouse(
                db_path       =f"data/warehouse_cs{chunk_size}.db",
                vector_db_path=f"data/vector_db_cs{chunk_size}",
            )
            warehouse.load_all(chunks)

            retriever = DocumentRetriever(warehouse)
            metrics   = self._evaluate_retrieval_metrics(retriever, k=3)

            # Quality metrics on first test query
            sample_query  = TEST_SUITE[0]['query']
            sample_chunks = retriever.get_relevant_chunks(sample_query, {'k': 3})
            response      = generator.generate_answer(sample_query, sample_chunks)
            quality       = QualityMetrics.composite_quality_score(
                response['answer'], sample_query, sample_chunks
            )

            result = {
                'experiment':       'chunk_size',
                'chunk_size':       chunk_size,
                'total_chunks':     len(chunks),
                'macro_precision':  metrics['macro_precision'],
                'macro_recall':     metrics['macro_recall'],
                'macro_f1':         metrics['macro_f1'],
                'MAP':              metrics['MAP'],
                'mean_ndcg':        metrics['mean_ndcg'],
                'quality_score':    quality['composite_score'],
                'timestamp':        datetime.now().isoformat(),
            }
            results.append(result)

            print(f"    Precision={metrics['macro_precision']:.4f}  "
                  f"Recall={metrics['macro_recall']:.4f}  "
                  f"F1={metrics['macro_f1']:.4f}  "
                  f"MAP={metrics['MAP']:.4f}  "
                  f"NDCG={metrics['mean_ndcg']:.4f}")

        self.results.extend(results)
        return results

    # ── Experiment 2: RAG vs No-RAG ───────────────────────────────────────────

    def run_rag_vs_no_rag_comparison(self, documents) -> List[Dict]:
        """Compare grounded RAG answers vs ungrounded keyword matching."""
        print("\n🧪  Experiment: RAG vs No-RAG")

        transformer = DataTransformer()
        chunks      = transformer.transform_pipeline(documents)
        warehouse   = DataWarehouse()
        warehouse.load_all(chunks)

        retriever = DocumentRetriever(warehouse)
        generator = AnswerGenerator()
        results   = []

        for case in TEST_SUITE:
            query = case['query']

            # RAG
            rag_chunks   = retriever.get_relevant_chunks(query, {'k': 3})
            rag_response = generator.generate_answer(query, rag_chunks)
            rag_quality  = QualityMetrics.composite_quality_score(
                rag_response['answer'], query, rag_chunks
            )

            # No-RAG — keyword matching over raw chunks
            no_rag_answer   = self._no_rag_answer(query, chunks)
            no_rag_quality  = QualityMetrics.composite_quality_score(
                no_rag_answer, query, []
            )

            # Retrieval metrics for RAG only
            raw_retrieved = retriever.retrieve_similar(query, k=3)
            retrieved_contents = [doc.page_content for doc, _ in raw_retrieved]
            p   = self.rm.precision_at_k(retrieved_contents, case['relevant'], 3)
            r   = self.rm.recall_at_k(retrieved_contents,   case['relevant'], 3)
            f1  = self.rm.f1_at_k(retrieved_contents,       case['relevant'], 3)

            result = {
                'experiment':          'rag_vs_no_rag',
                'query':               query,
                'rag_precision':       p,
                'rag_recall':          r,
                'rag_f1':              f1,
                'rag_quality_score':   rag_quality['composite_score'],
                'no_rag_quality_score':no_rag_quality['composite_score'],
                'timestamp':           datetime.now().isoformat(),
            }
            results.append(result)

            print(f"  [{query[:40]:40s}]  "
                  f"RAG F1={f1:.3f}  "
                  f"RAG Q={rag_quality['composite_score']:.3f}  "
                  f"NoRAG Q={no_rag_quality['composite_score']:.3f}")

        self.results.extend(results)
        return results

    def _no_rag_answer(self, query: str, all_chunks) -> str:
        """Simple keyword overlap — baseline without retrieval."""
        query_words = set(query.lower().split())
        best_chunk  = None
        max_overlap = 0

        for chunk in all_chunks:
            chunk_words = set(chunk.page_content.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_chunk  = chunk

        if best_chunk:
            return f"Based on course materials: {best_chunk.page_content[:500]}..."
        return "No relevant information found in study materials."

    # ── Save & Report ─────────────────────────────────────────────────────────

    def save_results(self, filename: str = "data/experiments_results.json"):
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n  Results saved → {filename}")

    def generate_report(self):
        if not self.results:
            print("No experiment results to report")
            return

        df = pd.DataFrame(self.results)
        print("\n📊  Experiment Results Summary:")
        print("  " + "─" * 60)

        for exp_name in df['experiment'].unique():
            sub = df[df['experiment'] == exp_name]
            print(f"\n  [{exp_name.upper()}]")

            if exp_name == 'chunk_size':
                for _, row in sub.iterrows():
                    print(f"    chunk={row['chunk_size']:5d}  "
                          f"P={row['macro_precision']:.4f}  "
                          f"R={row['macro_recall']:.4f}  "
                          f"F1={row['macro_f1']:.4f}  "
                          f"MAP={row['MAP']:.4f}  "
                          f"NDCG={row['mean_ndcg']:.4f}  "
                          f"Q={row['quality_score']:.4f}")

            elif exp_name == 'rag_vs_no_rag':
                print(f"    {'Query':40s}  RAG-F1  RAG-Q  NoRAG-Q")
                for _, row in sub.iterrows():
                    print(f"    {row['query'][:40]:40s}  "
                          f"{row['rag_f1']:.3f}   "
                          f"{row['rag_quality_score']:.3f}  "
                          f"{row['no_rag_quality_score']:.3f}")

        self.create_visualizations(df)

    def create_visualizations(self, df: pd.DataFrame):
        """Create and save experiment charts."""
        experiments = df['experiment'].unique()

        for exp in experiments:
            sub = df[df['experiment'] == exp]
            fig, axes = plt.subplots(1, 2 if exp == 'chunk_size' else 1,
                                     figsize=(14, 5))

            if exp == 'chunk_size':
                ax1, ax2 = axes
                ax1.plot(sub['chunk_size'], sub['macro_precision'], 'b-o', label='Precision')
                ax1.plot(sub['chunk_size'], sub['macro_recall'],    'r-o', label='Recall')
                ax1.plot(sub['chunk_size'], sub['macro_f1'],        'g-s', label='F1', linewidth=2)
                ax1.plot(sub['chunk_size'], sub['MAP'],             'm-^', label='MAP')
                ax1.set_xlabel('Chunk Size (words)')
                ax1.set_ylabel('Score')
                ax1.set_title('Retrieval Metrics vs Chunk Size')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1.05)

                ax2.bar(sub['chunk_size'].astype(str), sub['quality_score'], color='steelblue')
                ax2.set_xlabel('Chunk Size')
                ax2.set_ylabel('Answer Quality Score')
                ax2.set_title('Answer Quality vs Chunk Size')
                ax2.grid(True, alpha=0.3, axis='y')

            elif exp == 'rag_vs_no_rag':
                ax = axes if not isinstance(axes, (list, type(plt.axes()).__mro__[0])) else axes
                try:
                    ax = axes
                except Exception:
                    pass
                labels = [f"Q{i+1}" for i in range(len(sub))]
                x = range(len(labels))
                width = 0.35
                ax.bar([xi - width/2 for xi in x], sub['rag_quality_score'],
                       width, label='RAG', color='steelblue')
                ax.bar([xi + width/2 for xi in x], sub['no_rag_quality_score'],
                       width, label='No-RAG', color='coral')
                ax.set_xticks(list(x))
                ax.set_xticklabels(labels)
                ax.set_ylabel('Quality Score')
                ax.set_title('RAG vs No-RAG Answer Quality')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            out_path = f"data/experiment_{exp}.png"
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"  📈 Chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = ExperimentRunner()

    extractor = DataExtractor()
    documents = extractor.extract_from_directory("data")

    if not documents:
        print("❌  No documents found — run train_all.py first.")
    else:
        print(f"✅  Loaded {len(documents)} documents from data/")

        runner.run_chunk_size_experiment(documents)
        runner.run_rag_vs_no_rag_comparison(documents)
        runner.generate_report()
        runner.save_results()

        # Also run full evaluator
        print("\n🔍  Running full Evaluator report…")
        evaluator = Evaluator()
        evaluator.generate_evaluation_report(k=5)