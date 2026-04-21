# Analytics - Clustering Module (v2 — Fixed imports + variable name bug)
# Handles data mining and clustering analysis

from typing import Dict, List   # ← FIXED: was missing, caused NameError
import json
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import seaborn as sns

from etl.load import DataWarehouse


# ─────────────────────────────────────────────────────────────────────────────
# QueryAnalytics — base data access
# ─────────────────────────────────────────────────────────────────────────────

class QueryAnalytics:
    def __init__(self, warehouse: DataWarehouse = None):
        self.warehouse = warehouse or DataWarehouse()

    def get_query_data(self) -> pd.DataFrame:
        """Load all query log rows from the warehouse."""
        queries = self.warehouse.query_sql("SELECT * FROM queries")
        if not queries:
            return pd.DataFrame()

        df = pd.DataFrame(queries, columns=[
            'query_id', 'user_id', 'query_text', 'retrieved_chunks',
            'response', 'relevance_score', 'timestamp'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_chunk_data(self) -> pd.DataFrame:
        """Load all chunk rows from the warehouse."""
        chunks = self.warehouse.query_sql("SELECT * FROM chunks")
        if not chunks:
            return pd.DataFrame()

        df = pd.DataFrame(chunks, columns=[
            'chunk_id', 'doc_id', 'subject', 'topic', 'difficulty',
            'content', 'content_hash', 'word_count', 'source',
            'source_file', 'source_type', 'metadata', 'created_at'
        ])
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df


# ─────────────────────────────────────────────────────────────────────────────
# QueryClustering
# ─────────────────────────────────────────────────────────────────────────────

class QueryClustering:
    def __init__(self, analytics: QueryAnalytics):
        self.analytics = analytics

    def cluster_queries(self, n_clusters: int = 5) -> Dict:
        """Cluster user queries using K-means on TF-IDF features."""
        df = self.analytics.get_query_data()

        if df.empty or len(df) < n_clusters:
            return {'error': f'Not enough data for clustering '
                             f'(need ≥ {n_clusters} queries, have {len(df)})'}

        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['query_text'])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        df['cluster'] = clusters

        cluster_analysis: Dict = {}
        for cluster_id in range(n_clusters):
            cdf = df[df['cluster'] == cluster_id]
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size':           len(cdf),
                'avg_relevance':  float(cdf['relevance_score'].mean()) if 'relevance_score' in cdf else None,
                'sample_queries': cdf['query_text'].head(3).tolist(),
                'common_words':   self._get_common_words(cdf['query_text']),
            }

        return {
            'n_clusters':    n_clusters,
            'total_queries': len(df),
            'clusters':      cluster_analysis,
            'feature_names': vectorizer.get_feature_names_out().tolist(),
        }

    def _get_common_words(self, queries: pd.Series, top_n: int = 10) -> List[str]:
        stop = {'the','a','an','and','or','but','in','on','at','to','for','of',
                'with','by','is','are','was','were','what','how','why','when','where'}
        words = []
        for q in queries:
            words.extend(w for w in q.lower().split() if w not in stop and len(w) > 2)
        return [w for w, _ in Counter(words).most_common(top_n)]


# ─────────────────────────────────────────────────────────────────────────────
# ContentAnalytics
# ─────────────────────────────────────────────────────────────────────────────

class ContentAnalytics:
    def __init__(self, analytics: QueryAnalytics):
        self.analytics = analytics

    def topic_modeling(self, n_topics: int = 5) -> Dict:
        """LDA topic modeling on chunk content."""
        df = self.analytics.get_chunk_data()
        if df.empty:
            return {'error': 'No content data available'}

        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['content'])

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        topic_distributions = lda.fit_transform(X)

        feature_names = vectorizer.get_feature_names_out()
        topics: Dict = {}
        for idx, topic_vec in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic_vec.argsort()[:-11:-1]]
            topics[f'topic_{idx}'] = {
                'top_words':      top_words,
                'document_count': int((topic_distributions[:, idx] > 0.1).sum()),
            }

        return {
            'n_topics':            n_topics,
            'topics':              topics,
            'topic_distributions': topic_distributions.tolist(),
        }

    def find_frequent_patterns(self) -> Dict:
        """Classify query intent patterns."""
        df = self.analytics.get_query_data()
        if df.empty:
            return {'error': 'No query data available'}

        qt = df['query_text']
        patterns = {
            'concept_questions':    int(qt.str.contains(r'what is|explain|define',       case=False).sum()),
            'how_to_questions':     int(qt.str.contains(r'how to|how do',                 case=False).sum()),
            'comparison_questions': int(qt.str.contains(r'difference|compare|vs',        case=False).sum()),
            'code_questions':       int(qt.str.contains(r'code|program|function',        case=False).sum()),
            'troubleshooting':      int(qt.str.contains(r'error|problem|issue|wrong',    case=False).sum()),
        }

        return {
            'total_questions':    len(df),
            'patterns':           patterns,
            'most_common_starts': df['query_text'].str.lower().str.split().str[0].value_counts().head(10).to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# AssociationRuleMiner
# ─────────────────────────────────────────────────────────────────────────────

class AssociationRuleMiner:
    def __init__(self, analytics: QueryAnalytics):
        self.analytics = analytics

    def find_associations(self,
                          min_support: float = 0.1,
                          min_confidence: float = 0.5) -> List[Dict]:
        """Find co-occurrence associations between concepts in user sessions."""
        df = self.analytics.get_query_data()
        if df.empty:
            return []

        user_queries = df.groupby('user_id')['query_text'].apply(list).tolist()

        concepts = [
            'database', 'sql', 'join', 'normalization', 'indexing',
            'process', 'thread', 'memory', 'cpu', 'scheduling',
            'array', 'linked list', 'tree', 'graph', 'algorithm',
        ]

        total_users = max(len(user_queries), 1)
        associations: List[Dict] = []

        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i + 1:]:
                has_a = [concept_a in ' '.join(uq).lower() for uq in user_queries]
                has_b = [concept_b in ' '.join(uq).lower() for uq in user_queries]

                support      = sum(a and b for a, b in zip(has_a, has_b))
                support_ratio = support / total_users

                count_a = sum(has_a) or 1
                count_b = sum(has_b) or 1

                # FIXED: consistent variable names (was conf_a_to_b vs conf_a_b)
                conf_a_b = support / count_a
                conf_b_a = support / count_b

                if (support_ratio >= min_support and
                        (conf_a_b >= min_confidence or conf_b_a >= min_confidence)):
                    associations.append({
                        'antecedent':      concept_a,
                        'consequent':      concept_b,
                        'support':         round(support_ratio, 4),
                        'confidence_a_to_b': round(conf_a_b,    4),
                        'confidence_b_to_a': round(conf_b_a,    4),
                    })

        return sorted(associations, key=lambda x: x['support'], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Analytics Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticsDashboard:
    def __init__(self, analytics: QueryAnalytics):
        self.analytics         = analytics
        self.clustering        = QueryClustering(analytics)
        self.content_analytics = ContentAnalytics(analytics)
        self.association_miner = AssociationRuleMiner(analytics)

    def generate_dashboard_data(self) -> Dict:
        """Generate comprehensive analytics data for the dashboard."""
        query_df = self.analytics.get_query_data()
        chunk_df = self.analytics.get_chunk_data()

        if query_df.empty:
            return {
                'error': 'No query data available',
                'hint':  'Use the Chat or Exam modes to generate queries first.',
            }

        dashboard_data = {
            'overview': {
                'total_queries':     len(query_df),
                'unique_users':      int(query_df['user_id'].nunique()),
                'avg_relevance':     float(query_df['relevance_score'].mean())
                                     if 'relevance_score' in query_df else None,
                'total_chunks':      len(chunk_df) if not chunk_df.empty else 0,
                'subjects_covered':  int(chunk_df['subject'].nunique()) if not chunk_df.empty else 0,
            },
            'temporal_analysis':  self._temporal_analysis(query_df),
            'topic_analysis':     self._topic_analysis(query_df),
            'performance_metrics':self._performance_metrics(query_df),
            'clustering_results': self.clustering.cluster_queries(),
            'content_topics':     self.content_analytics.topic_modeling(),
            'patterns':           self.content_analytics.find_frequent_patterns(),
            'associations':       self.association_miner.find_associations(),
        }

        return dashboard_data

    def _temporal_analysis(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg({
            'query_id':       'count',
            'relevance_score': 'mean',
        }).reset_index()

        return {
            'daily_query_volume': daily.to_dict('records'),
            'peak_days':          daily.nlargest(5, 'query_id')['date'].astype(str).tolist(),
            'avg_daily_queries':  float(daily['query_id'].mean()),
        }

    def _topic_analysis(self, df: pd.DataFrame) -> Dict:
        topics_kw = {
            'Database':       ['database', 'sql', 'table', 'query', 'join', 'dbms'],
            'OS':             ['operating system', 'os', 'process', 'thread', 'memory', 'cpu'],
            'Data Structures':['array', 'linked list', 'tree', 'graph', 'algorithm'],
            'ML':             ['machine learning', 'neural', 'training', 'classification'],
            'Programming':    ['code', 'function', 'class', 'variable', 'loop'],
        }
        topic_counts: Dict = {}
        for topic, keywords in topics_kw.items():
            count = int(df['query_text'].str.lower().str.contains('|'.join(keywords)).sum())
            topic_counts[topic] = count

        return {
            'topic_popularity':   topic_counts,
            'most_popular_topic': max(topic_counts, key=topic_counts.get, default='N/A'),
        }

    def _performance_metrics(self, df: pd.DataFrame) -> Dict:
        scores = df['relevance_score'].dropna()
        return {
            'relevance_distribution': scores.describe().to_dict() if not scores.empty else {},
            'high_relevance_queries': int((scores > 0.8).sum()),
            'low_relevance_queries':  int((scores < 0.3).sum()),
            'avg_response_length':    float(df['response'].str.len().mean())
                                      if 'response' in df else 0.0,
        }


if __name__ == "__main__":
    analytics  = QueryAnalytics()
    dashboard  = AnalyticsDashboard(analytics)
    data       = dashboard.generate_dashboard_data()
    print(json.dumps(data, indent=2, default=str))