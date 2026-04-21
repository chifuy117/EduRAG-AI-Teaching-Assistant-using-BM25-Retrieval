# Main Application Entry Point
# This module provides the main entry point for the AI Teaching Assistant

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataWarehouse
from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator
import json

def setup_system():
    """Initialize the complete system"""
    print("🚀 Initializing AI Teaching Assistant System...")

    # Initialize components
    warehouse = DataWarehouse()
    retriever = DocumentRetriever(warehouse)
    generator = AnswerGenerator()

    print("✅ System initialized successfully!")
    return warehouse, retriever, generator

def show_processing_stats(chunks):
    """Show processing statistics"""
    from collections import Counter

    subjects = Counter()
    topics = Counter()
    sources = Counter()

    for chunk in chunks:
        metadata = chunk.metadata
        subjects[metadata.get('subject', 'unknown')] += 1
        topics[metadata.get('topic', 'unknown')] += 1
        sources[metadata.get('source_file', 'unknown')] += 1

    print("\n📊 Processing Statistics:")
    print(f"Subjects: {dict(subjects)}")
    print(f"Top topics: {dict(topics.most_common(5))}")
    print(f"Source files: {dict(sources)}")
    """Show processing statistics"""
    from collections import Counter

    subjects = Counter()
    topics = Counter()
    sources = Counter()

    for chunk in chunks:
        metadata = chunk.metadata
        subjects[metadata.get('subject', 'unknown')] += 1
        topics[metadata.get('topic', 'unknown')] += 1
        sources[metadata.get('source_file', 'unknown')] += 1

    print("\n📊 Processing Statistics:")
    print(f"Subjects: {dict(subjects)}")
    print(f"Top topics: {dict(topics.most_common(5))}")
    print(f"Source files: {dict(sources)}")

def test_rag_system(query="What is normalization in databases?", subject=None):
    """Test the RAG system with a sample query"""
    print(f"🧪 Testing RAG system with query: '{query}'")

    from syllabus import get_subject_from_query, validate_topic_in_syllabus

    warehouse, retriever, generator = setup_system()

    # Detect subject from query if not provided
    detected_subject = subject or get_subject_from_query(query)
    if detected_subject:
        print(f"🎯 Detected subject: {detected_subject}")

    # Prepare filters
    filters = {'k': 3}
    if detected_subject:
        filters['subject'] = detected_subject

    # Retrieve relevant chunks
    chunks = retriever.get_relevant_chunks(query, filters)
    print(f"📚 Retrieved {len(chunks)} relevant chunks")

    # Generate answer
    response = generator.generate_answer(query, chunks)
    print("🤖 Generated answer:")
    print(response['answer'])

    # Syllabus validation
    if detected_subject:
        topic_in_syllabus = validate_topic_in_syllabus(detected_subject, query)
        print(f"📋 Topic in syllabus: {topic_in_syllabus}")

    # Evaluate
    evaluation = generator.evaluate_answer_quality(response['answer'], query, chunks)
    print("📊 Answer quality evaluation:")
    print(json.dumps(evaluation, indent=2))

    return response

def show_system_info():
    """Show system information and statistics"""
    print("ℹ️  AI Teaching Assistant System Info")
    print("=" * 50)

    warehouse = DataWarehouse()

    # Query statistics
    query_count = warehouse.query_sql("SELECT COUNT(*) FROM queries")[0][0]
    print(f"Total Queries: {query_count}")

    # Chunk statistics
    chunk_count = warehouse.query_sql("SELECT COUNT(*) FROM chunks")[0][0]
    print(f"Total Content Chunks: {chunk_count}")

    # Subject breakdown
    subjects = warehouse.query_sql("SELECT subject, COUNT(*) FROM chunks GROUP BY subject")
    print("Content by Subject:")
    for subject, count in subjects:
        print(f"  - {subject}: {count} chunks")

    # Recent activity
    recent_queries = warehouse.query_sql("SELECT timestamp, query_text FROM queries ORDER BY timestamp DESC LIMIT 5")
    if recent_queries:
        print("Recent Queries:")
        for timestamp, query in recent_queries:
            print(f"  - {timestamp}: {query[:50]}...")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Teaching Assistant")
    parser.add_argument('--setup', action='store_true', help='Setup the system')
    parser.add_argument('--test-rag', action='store_true', help='Test RAG system')
    parser.add_argument('--test-query', type=str, default="What is normalization in databases?", help='Test query')
    parser.add_argument('--info', action='store_true', help='Show system info')

    args = parser.parse_args()

    if args.setup:
        setup_system()

    elif args.test_rag:
        test_rag_system(query=args.test_query)

    elif args.info:
        show_system_info()

    else:
        print("🤖 AI Teaching Assistant")
        print("=" * 30)
        print("Usage examples:")
        print("  python main.py --setup                    # Initialize system")
        print("  python main.py --test-rag                 # Test RAG system")
        print("  python main.py --test-rag --test-query 'What is BFS?'  # Custom test query")
        print("  python main.py --info                     # Show system info")
        print("  python train_all.py                       # Full training pipeline")
        print("  python launcher.py                        # Launch UI")
        print()

if __name__ == "__main__":
    main()