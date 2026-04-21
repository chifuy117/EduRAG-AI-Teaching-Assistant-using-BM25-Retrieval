# UI Module - Streamlit Web Interface
# This module provides the web interface for the AI Teaching Assistant

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
import re
from datetime import datetime
from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataWarehouse
from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator
from modes.quiz import QuizGenerator
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from syllabus import get_subject_from_query, validate_topic_in_syllabus, SYLLABUS

# Page configuration
st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all system components"""
    warehouse = DataWarehouse()
    retriever = DocumentRetriever(warehouse)
    generator = AnswerGenerator()
    quiz_gen = QuizGenerator(generator=generator)
    return warehouse, retriever, generator, quiz_gen

warehouse, retriever, generator, quiz_gen = init_components()

def normalize_subject_query(subject: str) -> str:
    """Convert labels like MachineLearning to human-readable search text."""
    if not subject or subject == 'All':
        return ''
    subject_text = subject.replace('_', ' ')
    subject_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', subject_text)
    return subject_text.strip()


def main():
    """Main application"""
    st.title("🎓 AI Teaching Assistant")
    st.markdown("Your intelligent learning companion powered by RAG and Data Mining")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Subject selection
        subjects = ["All"] + list(SYLLABUS.keys())
        selected_subject = st.selectbox("Select Subject", subjects)

        # Difficulty level
        difficulties = ["All", "Easy", "Medium", "Hard"]
        selected_difficulty = st.selectbox("Difficulty Level", difficulties)

        # Advanced options
        with st.expander("Advanced Options"):
            max_chunks = st.slider("Max Retrieved Chunks", 1, 10, 3)
            show_sources = st.checkbox("Show Sources", True)
            show_metadata = st.checkbox("Show Metadata", False)

        # Analytics button
        if st.button("📊 View Analytics Dashboard"):
            st.session_state.page = "analytics"

    # Main content area
    if 'page' not in st.session_state:
        st.session_state.page = "chat"

    if st.session_state.page == "chat":
        chat_interface(selected_subject, selected_difficulty, max_chunks, show_sources, show_metadata)
    elif st.session_state.page == "analytics":
        analytics_dashboard()

def chat_interface(subject, difficulty, max_chunks, show_sources, show_metadata):
    """Main chat interface"""
    st.header("💬 Ask Your Question")

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
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Detect subject from query
                detected_subject = get_subject_from_query(prompt)

                # Prepare filters
                filters = {
                    'k': max_chunks
                }

                # Use selected subject or detected subject
                query_subject = subject if subject != "All" else (detected_subject or "general")
                if query_subject != "general":
                    filters['subject'] = query_subject.lower()

                if difficulty != "All":
                    filters['difficulty'] = difficulty.lower()

                # Syllabus validation
                syllabus_check = None
                if detected_subject:
                    syllabus_check = validate_topic_in_syllabus(detected_subject, prompt)

                # Retrieve relevant chunks
                chunks = retriever.get_relevant_chunks(prompt, filters)

                # Generate answer
                response = generator.generate_answer(prompt, chunks)

                # Display answer
                st.write(response['answer'])

                # Show syllabus validation
                if syllabus_check is not None:
                    if syllabus_check:
                        st.success("✅ This topic is covered in your syllabus")
                    else:
                        st.warning("⚠️ This topic may not be in your current syllabus")

                # Show subject detection
                if detected_subject and detected_subject != query_subject:
                    st.info(f"🎯 Detected subject: {detected_subject}")

                # Show sources if enabled
                if show_sources and 'sources' in response:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(response['sources'], 1):
                            st.write(f"{i}. {source}")

                # Show metadata if enabled
                if show_metadata and chunks:
                    with st.expander("📋 Metadata"):
                        for i, chunk in enumerate(chunks, 1):
                            st.write(f"**Chunk {i}:**")
                            st.json(chunk['metadata'])

                # Log query for analytics
                warehouse.log_query(
                    user_id="web_user",  # In real app, use actual user ID
                    query_text=prompt,
                    retrieved_chunks=[chunk['content'][:200] for chunk in chunks],
                    response=response['answer'],
                    relevance_score=sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks) if chunks else 0
                )

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response.get('sources', [])
        })


def quiz_interface(subject, difficulty):
    """Quiz generation interface"""
    st.header("📝 Quiz Generator")
    st.markdown("Generate a quiz from your course materials and test your knowledge.")

    subjects = ["All"] + list(SYLLABUS.keys())
    selected_subject = st.selectbox("Select Subject", subjects, index=subjects.index(subject) if subject in subjects else 0)

    difficulties = ["All", "Easy", "Medium", "Hard"]
    selected_difficulty = st.selectbox("Difficulty Level", difficulties, index=difficulties.index(difficulty) if difficulty in difficulties else 0)

    topic_prompt = st.text_input(
        "Quiz Topic / Question Focus",
        value=selected_subject if selected_subject != "All" else "",
        help="Enter a topic, keyword or question that should guide the quiz generation."
    )

    quiz_type = st.radio(
        "Quiz Type",
        ["MCQ", "Short Answer", "Mixed"],
        horizontal=True
    )

    num_questions = st.slider("Number of Questions", 1, 15, 5)

    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None

    if st.button("Generate Quiz", use_container_width=True):
        if not topic_prompt and selected_subject == "All":
            st.warning("Enter a quiz topic or select a specific subject to generate a quiz.")
        else:
            with st.spinner("Generating quiz from course materials..."):
                query_text = topic_prompt.strip() or normalize_subject_query(selected_subject)
                filters = {'k': max(10, num_questions * 3)}
                if selected_subject != "All":
                    filters['subject'] = selected_subject
                if selected_difficulty != "All":
                    filters['difficulty'] = selected_difficulty.lower()

                chunks = retriever.get_relevant_chunks(query_text, filters)

                if not chunks:
                    st.error(
                        "No course material chunks were found for this topic. "
                        "Please upload or process documents first in Data Management."
                    )
                    return

                if quiz_type == "MCQ":
                    quiz = quiz_gen.generate_mcq_quiz(chunks, num_questions)
                elif quiz_type == "Short Answer":
                    quiz = quiz_gen.generate_short_answer_quiz(chunks, num_questions)
                else:
                    quiz = quiz_gen.generate_mixed_quiz(chunks, num_questions)

                st.session_state.current_quiz = quiz
                st.session_state.quiz_topic = query_text
                st.session_state.quiz_type = quiz_type
                st.session_state.quiz_subject = selected_subject
                st.session_state.quiz_difficulty = selected_difficulty

    if st.session_state.current_quiz:
        quiz = st.session_state.current_quiz
        st.subheader(f"Quiz: {quiz.get('quiz_type', 'Mixed').title()} — {quiz.get('total_questions', 0)} questions")
        st.info(f"Topic: {st.session_state.get('quiz_topic', 'General')}")
        st.write(f"**Estimated duration:** {quiz.get('duration_minutes', 0)} minutes")

        responses = []
        for question in quiz.get('questions', []):
            st.divider()
            st.write(f"**Q{question['question_number']}:** {question['question']}")
            if question['type'] == 'mcq':
                response = st.radio(
                    f"Answer Q{question['question_number']}",
                    question['options'],
                    key=f"quiz_q_{question['question_number']}"
                )
            else:
                response = st.text_area(
                    f"Answer Q{question['question_number']}",
                    key=f"quiz_q_{question['question_number']}"
                )
            responses.append(response)

        if st.button("Submit Quiz", use_container_width=True):
            quiz = st.session_state.current_quiz
            response_values = [
                st.session_state.get(f"quiz_q_{q['question_number']}", "")
                for q in quiz.get('questions', [])
            ]
            results = quiz_gen.evaluate_quiz_response(quiz, response_values)
            st.success("Quiz submitted!")
            st.metric("Score", f"{results['score']}/{results['total_marks']}")
            st.metric("Percentage", results['percentage'])
            st.write(results['feedback'])

            with st.expander("Detailed Results"):
                for result in results['results']:
                    status = "✓" if result['is_correct'] else "✗"
                    st.write(f"{status} Q{result['question_number']}: {result['marks_obtained']}/{result['marks_total']} — {result['correct_answer']}")

            warehouse.log_query(
                user_id="quiz_user",
                query_text=st.session_state.get('quiz_topic', ''),
                retrieved_chunks=[q.get('question', '') for q in quiz.get('questions', [])],
                response=results['feedback'],
                relevance_score=0.0
            )


def analytics_dashboard():
    """Analytics dashboard"""
    st.header("📊 Analytics Dashboard")

    # Get analytics data
    queries = warehouse.query_sql("SELECT * FROM queries ORDER BY timestamp DESC LIMIT 100")

    if not queries:
        st.info("No query data available yet. Start asking questions to see analytics!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(queries, columns=['query_id', 'user_id', 'query_text', 'retrieved_chunks', 'response', 'relevance_score', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", len(df))

    with col2:
        avg_relevance = df['relevance_score'].mean()
        st.metric("Avg Relevance Score", ".3f")

    with col3:
        unique_users = df['user_id'].nunique()
        st.metric("Active Users", unique_users)

    with col4:
        today_queries = df[df['timestamp'].dt.date == datetime.now().date()].shape[0]
        st.metric("Queries Today", today_queries)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Query Volume Over Time")
        daily_queries = df.groupby(df['timestamp'].dt.date).size().reset_index(name='count')
        fig = px.line(daily_queries, x='timestamp', y='count', title="Daily Query Volume")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Relevance Score Distribution")
        fig = px.histogram(df, x='relevance_score', title="Relevance Scores")
        st.plotly_chart(fig, use_container_width=True)

    # Most asked topics (simple keyword analysis)
    st.subheader("🔥 Most Asked Topics")

    all_queries = ' '.join(df['query_text'].str.lower())
    topics = {
        'Database': ['database', 'sql', 'table', 'query', 'join', 'dbms'],
        'OS': ['operating system', 'process', 'thread', 'memory', 'cpu'],
        'Data Structures': ['array', 'linked list', 'tree', 'graph', 'algorithm'],
        'Programming': ['code', 'function', 'class', 'variable', 'loop']
    }

    topic_counts = {}
    for topic, keywords in topics.items():
        count = sum(all_queries.count(keyword) for keyword in keywords)
        topic_counts[topic] = count

    topic_df = pd.DataFrame(list(topic_counts.items()), columns=['Topic', 'Mentions'])
    fig = px.bar(topic_df, x='Topic', y='Mentions', title="Topic Popularity")
    st.plotly_chart(fig, use_container_width=True)

    # Recent queries table
    st.subheader("📝 Recent Queries")
    recent_df = df[['timestamp', 'query_text', 'relevance_score']].head(10)
    st.dataframe(recent_df)

def data_management_page():
    """Data management interface"""
    st.header("📁 Data Management")

    tab1, tab2 = st.tabs(["Upload Documents", "Process Data"])

    with tab1:
        st.subheader("Upload PDF Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                extractor = DataExtractor()
                transformer = DataTransformer()

                all_docs = []
                for uploaded_file in uploaded_files:
                    # Save temporarily
                    with open(f"temp_{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract
                    docs = extractor.extract_from_pdf(f"temp_{uploaded_file.name}")
                    all_docs.extend(docs)

                    # Clean up
                    os.remove(f"temp_{uploaded_file.name}")

                # Transform and load
                chunks = transformer.transform_pipeline(all_docs)
                warehouse.load_all(chunks)

                st.success(f"Processed {len(uploaded_files)} PDFs with {len(chunks)} chunks!")

    with tab2:
        st.subheader("Data Processing Status")

        # Show current data stats
        chunks_count = warehouse.query_sql("SELECT COUNT(*) FROM chunks")[0][0]
        subjects = warehouse.query_sql("SELECT DISTINCT subject FROM chunks")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Chunks", chunks_count)

        with col2:
            st.metric("Subjects", len(subjects))

        if st.button("Rebuild Vector Database"):
            with st.spinner("Rebuilding vector database..."):
                # Get all chunks
                chunks_data = warehouse.query_sql("SELECT content, metadata FROM chunks")
                chunks = []
                for content, metadata_str in chunks_data:
                    metadata = json.loads(metadata_str)
                    from langchain.schema import Document
                    doc = Document(page_content=content, metadata=metadata)
                    chunks.append(doc)

                # Rebuild vector store
                warehouse.load_to_vector_db(chunks)
                st.success("Vector database rebuilt!")

if __name__ == "__main__":
    # Add navigation
    st.sidebar.markdown("---")
    pages = ["Chat", "Quiz", "Analytics", "Data Management"]
    page_choice = st.sidebar.radio("Navigation", pages)

    if page_choice == "Chat":
        st.session_state.page = "chat"
    elif page_choice == "Quiz":
        st.session_state.page = "quiz"
    elif page_choice == "Analytics":
        st.session_state.page = "analytics"
    elif page_choice == "Data Management":
        st.session_state.page = "data_management"

    if st.session_state.get('page') == "data_management":
        data_management_page()
    elif st.session_state.get('page') == "quiz":
        quiz_interface("All", "All")
    else:
        main()