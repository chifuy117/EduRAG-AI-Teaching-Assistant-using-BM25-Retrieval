# Updated UI Module - Streamlit Web Interface
# Complete interface with all features: Learning Mode, Exam Mode, Quiz, Analytics

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
from datetime import datetime
from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataWarehouse
from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator
from modes.quiz import QuizGenerator
from syllabus import get_subject_from_query, validate_topic_in_syllabus, SYLLABUS
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="AI Teaching Assistant Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text.align: center;
    }
    .mode-selector {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all system components"""
    warehouse = DataWarehouse()
    retriever = DocumentRetriever(warehouse)
    generator = AnswerGenerator()
    quiz_gen = QuizGenerator()
    return warehouse, retriever, generator, quiz_gen

warehouse, retriever, generator, quiz_gen = init_components()

def main():
    """Main application"""
    st.title("🎓 AI Teaching Assistant Pro")
    st.markdown("Complete learning and exam preparation system powered by RAG, Data Mining & Analytics")

    # Sidebar navigation
    with st.sidebar:
        st.header("⚙️ Navigation")

        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["💬 Chat", "� Quiz", "📈 Analytics"],
            key="mode_selection"
        )

        st.divider()

        # Subject selection
        st.subheader("📖 Subject")
        subjects = ["All"] + list(SYLLABUS.keys())
        selected_subject = st.selectbox("Select Subject", subjects)

        # Difficulty level
        st.subheader("⚡ Difficulty")
        difficulties = ["All", "Easy", "Medium", "Hard"]
        selected_difficulty = st.selectbox("Difficulty Level", difficulties)

        st.divider()

        # Advanced options
        st.subheader("🔧 Advanced")
        max_chunks = st.slider("Max Retrieved Chunks", 1, 10, 3)
        show_sources = st.checkbox("Show Sources", True)
        show_metadata = st.checkbox("Show Metadata", False)
        show_confidence = st.checkbox("Show Confidence Scores", True)

    # Route to different modes
    if mode == "💬 Chat":
        chat_mode(selected_subject, selected_difficulty, max_chunks, show_sources, show_metadata)

    elif mode == "� Quiz":
        quiz_interface(selected_subject, selected_difficulty)

    elif mode == "📈 Analytics":
        analytics_dashboard()

def chat_mode(subject, difficulty, max_chunks, show_sources, show_metadata):
    """Standard chat interface"""
    st.header("💬 Chat with Tutor")

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                if show_sources and message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"{i}. {source}")

    # Chat input
    if prompt := st.chat_input("Ask me anything about your courses..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                detected_subject = get_subject_from_query(prompt)

                filters = {'k': max_chunks}
                query_subject = subject if subject != "All" else (detected_subject or "general")
                if query_subject != "general":
                    filters['subject'] = query_subject.lower()
                if difficulty != "All":
                    filters['difficulty'] = difficulty.lower()

                chunks = retriever.get_relevant_chunks(prompt, filters)
                response = generator.generate_answer(prompt, chunks)

                st.write(response['answer'])

                if show_sources and 'sources' in response:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(response['sources'], 1):
                            st.write(f"{i}. {source}")

                if show_metadata and chunks:
                    with st.expander("📋 Metadata"):
                        for i, chunk in enumerate(chunks, 1):
                            st.write(f"**Chunk {i}:**")
                            st.json(chunk['metadata'])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response.get('sources', [])
        })

def quiz_interface(subject, difficulty):
    """Quiz generation and evaluation"""
    st.header("📊 Quiz Generator")

    quiz_type = st.radio(
        "Quiz Type",
        ["MCQ", "Short Answer", "Mixed"],
        horizontal=True
    )

    num_questions = st.slider("Number of Questions", 1, 20, 5)

    if st.button("Generate Quiz", use_container_width=True):
        with st.spinner("Generating quiz..."):
            # Get chunks
            extractor = DataExtractor()
            chunks = extractor.extract_from_directory("data") if os.path.exists("data") else []

            if not chunks:
                st.error("No content available. Please upload course materials first.")
                return

            # Filter by subject and difficulty
            if subject != "All":
                chunks = [c for c in chunks if c['metadata'].get('subject', '').lower() == subject.lower()]
            if difficulty != "All":
                chunks = [c for c in chunks if c['metadata'].get('difficulty', '').lower() == difficulty.lower()]

            # Generate quiz
            if quiz_type == "MCQ":
                quiz = quiz_gen.generate_mcq_quiz(chunks, num_questions)
            elif quiz_type == "Short Answer":
                quiz = quiz_gen.generate_short_answer_quiz(chunks, num_questions)
            else:
                quiz = quiz_gen.generate_mixed_quiz(chunks, num_questions)

            # Display quiz
            st.subheader(f"Quiz: {quiz_type} ({quiz['total_questions']} questions)")
            st.info(f"⏱️ Estimated Duration: {quiz['duration_minutes']} minutes")

            # Quiz questions
            responses = []
            for q in quiz['questions']:
                st.divider()
                st.write(f"**Q{q['question_number']}:** {q['question']}")

                if q['type'] == 'mcq':
                    response = st.radio(
                        f"Select answer for Q{q['question_number']}",
                        q['options'],
                        key=f"q_{q['question_number']}"
                    )
                else:
                    response = st.text_area(
                        f"Answer Q{q['question_number']}",
                        key=f"q_{q['question_number']}"
                    )

                responses.append(response)

            # Evaluate
            if st.button("Submit Quiz", use_container_width=True):
                results = quiz_gen.evaluate_quiz_response(quiz, responses)

                st.success("Quiz Submitted!")
                st.metric("Score", f"{results['score']}/{results['total_marks']}")
                st.metric("Percentage", results['percentage'])

                st.info(results['feedback'])

                with st.expander("Detailed Results"):
                    for result in results['results']:
                        status = "✓" if result['is_correct'] else "✗"
                        st.write(f"{status} Q{result['question_number']}: {result['marks_obtained']}/{result['marks_total']} marks")

def analytics_dashboard():
    """Analytics and insights"""
    st.header("📈 Analytics Dashboard")

    # Get data
    query_data = warehouse.query_sql("SELECT * FROM queries LIMIT 100")

    if not query_data:
        st.info("No query data available yet. Start using the system to generate analytics.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(query_data, columns=[
        'query_id', 'user_id', 'query_text', 'retrieved_chunks',
        'response', 'relevance_score', 'timestamp'
    ])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", len(df))
    with col2:
        st.metric("Unique Users", df['user_id'].nunique())
    with col3:
        st.metric("Avg Relevance", f"{df['relevance_score'].mean():.2f}" if 'relevance_score' in df.columns else "N/A")
    with col4:
        st.metric("Date Range", f"{df['timestamp'].min()[:10] if len(df) > 0 else 'N/A'}")

def syllabus_interface(subject):
    """Syllabus analysis interface"""
    st.header("🎯 Syllabus Management")

    tabs = st.tabs(["Coverage Analysis", "Topic Validation", "Study Guide"])

    with tabs[0]:
        st.subheader("📊 Coverage Analysis")

        if subject != "All":
            if subject in SYLLABUS:
                topics = SYLLABUS[subject]['topics']
                st.write(f"**{subject} Syllabus Topics ({len(topics)})**")

                cols = st.columns(3)
                for i, topic in enumerate(topics):
                    with cols[i % 3]:
                        st.write(f"• {topic}")

    with tabs[1]:
        st.subheader("✅ Topic Validation")
        query = st.text_input("Enter a topic to validate:")

        if query and subject != "All":
            validation = syllabus_val.validate_topic(subject, query)
            if validation['valid']:
                st.success(validation['message'])
            else:
                st.warning(validation['message'])

    with tabs[2]:
        st.subheader("📚 Study Guide")
        if subject != "All":
            extractor = DataExtractor()
            chunks = extractor.extract_from_directory("data") if os.path.exists("data") else []

            if chunks:
                guide = syllabus_val.generate_study_guide(subject, chunks)
                if 'study_units' in guide:
                    guide_df = pd.DataFrame(guide['study_units'])
                    st.dataframe(guide_df, use_container_width=True)

if __name__ == "__main__":
    main()