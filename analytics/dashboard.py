# Analytics - Dashboard Module
# This module provides the analytics dashboard functionality

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from analytics.clustering import AnalyticsDashboard, QueryAnalytics
from etl.load import DataWarehouse
import json

class StreamlitDashboard:
    def __init__(self):
        self.warehouse = DataWarehouse()
        self.analytics = QueryAnalytics(self.warehouse)
        self.dashboard = AnalyticsDashboard(self.analytics)

    def display_overview_metrics(self, data):
        """Display overview metrics"""
        st.header("📊 Overview Metrics")

        overview = data.get('overview', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", overview.get('total_queries', 0))

        with col2:
            st.metric("Active Users", overview.get('unique_users', 0))

        with col3:
            st.metric("Avg Relevance", ".3f")

        with col4:
            st.metric("Content Chunks", overview.get('total_chunks', 0))

    def display_temporal_analysis(self, data):
        """Display temporal analysis charts"""
        st.header("📈 Temporal Analysis")

        temporal = data.get('temporal_analysis', {})

        col1, col2 = st.columns(2)

        with col1:
            daily_data = temporal.get('daily_query_volume', [])
            if daily_data:
                df = pd.DataFrame(daily_data)
                fig = px.line(df, x='date', y='query_id',
                            title="Daily Query Volume",
                            labels={'query_id': 'Queries', 'date': 'Date'})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if daily_data:
                df = pd.DataFrame(daily_data)
                fig = px.line(df, x='date', y='relevance_score',
                            title="Daily Average Relevance",
                            labels={'relevance_score': 'Relevance Score', 'date': 'Date'})
                st.plotly_chart(fig, use_container_width=True)

    def display_topic_analysis(self, data):
        """Display topic analysis"""
        st.header("🎯 Topic Analysis")

        topic_data = data.get('topic_analysis', {}).get('topic_popularity', {})

        if topic_data:
            df = pd.DataFrame(list(topic_data.items()), columns=['Topic', 'Queries'])
            fig = px.bar(df, x='Topic', y='Queries',
                        title="Topic Popularity",
                        color='Queries',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)

            # Most popular topic
            most_popular = data.get('topic_analysis', {}).get('most_popular_topic', 'N/A')
            st.info(f"📈 Most Popular Topic: **{most_popular}**")

    def display_performance_metrics(self, data):
        """Display performance metrics"""
        st.header("⚡ Performance Metrics")

        perf = data.get('performance_metrics', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            relevance_dist = perf.get('relevance_distribution', {})
            st.metric("Median Relevance", ".3f")

        with col2:
            st.metric("High Relevance Queries (>0.8)",
                     perf.get('high_relevance_queries', 0))

        with col3:
            st.metric("Low Relevance Queries (<0.3)",
                     perf.get('low_relevance_queries', 0))

        # Relevance score distribution
        if 'relevance_distribution' in perf:
            st.subheader("Relevance Score Distribution")
            dist_data = perf['relevance_distribution']

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=list(dist_data.values()),
                nbinsx=20,
                name="Relevance Scores"
            ))
            fig.update_layout(
                title="Distribution of Relevance Scores",
                xaxis_title="Relevance Score",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)

    def display_clustering_results(self, data):
        """Display clustering analysis"""
        st.header("🔍 Query Clustering")

        clustering = data.get('clustering_results', {})

        if 'error' in clustering:
            st.warning(clustering['error'])
            return

        n_clusters = clustering.get('n_clusters', 0)
        st.info(f"📊 Found {n_clusters} distinct query clusters")

        clusters = clustering.get('clusters', {})

        for cluster_id, cluster_info in clusters.items():
            with st.expander(f"Cluster {cluster_id.split('_')[1]} - {cluster_info['size']} queries"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Cluster Size", cluster_info['size'])
                    st.metric("Avg Relevance", ".3f")

                with col2:
                    st.write("**Common Words:**")
                    st.write(", ".join(cluster_info.get('common_words', [])))

                st.write("**Sample Queries:**")
                for query in cluster_info.get('sample_queries', []):
                    st.write(f"• {query}")

    def display_content_topics(self, data):
        """Display content topic modeling"""
        st.header("📚 Content Topics")

        topics_data = data.get('content_topics', {})

        if 'error' in topics_data:
            st.warning(topics_data['error'])
            return

        topics = topics_data.get('topics', {})

        for topic_id, topic_info in topics.items():
            with st.expander(f"Topic {topic_id.split('_')[1]}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Documents", topic_info.get('document_count', 0))

                with col2:
                    st.write("**Top Words:**")
                    st.write(", ".join(topic_info.get('top_words', [])))

    def display_patterns_and_associations(self, data):
        """Display patterns and associations"""
        st.header("🔗 Patterns & Associations")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Question Patterns")
            patterns = data.get('patterns', {})

            if 'patterns' in patterns:
                pattern_data = patterns['patterns']
                df = pd.DataFrame(list(pattern_data.items()), columns=['Pattern', 'Count'])
                fig = px.bar(df, x='Pattern', y='Count',
                            title="Question Pattern Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Association Rules")
            associations = data.get('associations', [])

            if associations:
                # Display top associations
                for assoc in associations[:5]:
                    st.write(f"**{assoc['antecedent']}** → **{assoc['consequent']}**")
                    st.write(".3f")
                    st.write("---")
            else:
                st.info("No strong associations found yet")

    def display_raw_data(self, data):
        """Display raw data tables"""
        st.header("📋 Raw Data")

        tab1, tab2 = st.tabs(["Recent Queries", "Content Chunks"])

        with tab1:
            query_df = self.analytics.get_query_data()
            if not query_df.empty:
                # Show last 20 queries
                recent_queries = query_df.tail(20)[['timestamp', 'query_text', 'relevance_score']]
                st.dataframe(recent_queries)
            else:
                st.info("No query data available")

        with tab2:
            chunk_df = self.analytics.get_chunk_data()
            if not chunk_df.empty:
                # Show chunk summary
                chunk_summary = chunk_df[['chunk_id', 'subject', 'topic', 'difficulty', 'word_count', 'source']]
                st.dataframe(chunk_summary)
            else:
                st.info("No content data available")

    def run_dashboard(self):
        """Run the complete dashboard"""
        st.title("📊 AI Teaching Assistant Analytics")

        # Generate dashboard data
        with st.spinner("Generating analytics..."):
            data = self.dashboard.generate_dashboard_data()

        if 'error' in data:
            st.error(f"Error generating analytics: {data['error']}")
            return

        # Display all sections
        self.display_overview_metrics(data)
        st.markdown("---")

        self.display_temporal_analysis(data)
        st.markdown("---")

        self.display_topic_analysis(data)
        st.markdown("---")

        self.display_performance_metrics(data)
        st.markdown("---")

        self.display_clustering_results(data)
        st.markdown("---")

        self.display_content_topics(data)
        st.markdown("---")

        self.display_patterns_and_associations(data)
        st.markdown("---")

        self.display_raw_data(data)

        # Export functionality
        st.header("💾 Export Data")
        if st.button("Export Analytics Report (JSON)"):
            json_data = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="analytics_report.json",
                mime="application/json"
            )

if __name__ == "__main__":
    dashboard = StreamlitDashboard()
    dashboard.run_dashboard()