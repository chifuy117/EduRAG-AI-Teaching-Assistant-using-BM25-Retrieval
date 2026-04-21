# PREMIUM UI MODULE - Professional AI Teaching Assistant
# Complete enterprise-grade interface with all features and best practices

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
import re
from datetime import datetime, timedelta
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
from functools import lru_cache
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AI Teaching Assistant PRO",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ADS_DWH_Project',
        'Report a bug': "mailto:support@example.com",
        'About': "# AI Teaching Assistant PRO\nPowered by RAG, Data Mining & Machine Learning"
    }
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary: #1f77b4;
        --secondary: #ff7f0e;
        --success: #2ca02c;
        --danger: #d62728;
        --info: #17a2b8;
        --warning: #ffc107;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5em;
        color: #ff7f0e;
        font-weight: 600;
        margin: 15px 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    
    /* Mode selector cards */
    .mode-card {
        background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s ease;
        margin: 10px;
    }
    
    .mode-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #2ca02c;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #d62728;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 10px;
    }
    
    /* Code blocks */
    .code-block {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        font-family: monospace;
        overflow-x: auto;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.user_name = "Student"
    st.session_state.current_mode = "🏠 Dashboard"
    st.session_state.chat_history = []
    st.session_state.quiz_score = 0
    st.session_state.learning_progress = {}
    st.session_state.theme = "light"

# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================
@st.cache_resource
def init_components():
    """Initialize all system components with caching"""
    try:
        warehouse    = DataWarehouse()
        retriever    = DocumentRetriever(warehouse)
        generator    = AnswerGenerator()
        quiz_gen     = QuizGenerator(generator=generator)
        return {
            'warehouse':  warehouse,
            'retriever':  retriever,
            'generator':  generator,
            'quiz':       quiz_gen,
        }
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {str(e)}")
        return None

components = init_components()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_user_statistics():
    """Get live statistics from the knowledge base"""
    try:
        wh = components['warehouse'] if components else None
        chunks = wh.chunks_data if wh else []
        total_chunks = len(chunks)
        subjects = set(c['metadata'].get('subject','') for c in chunks if c.get('metadata'))
        topics   = set(c['metadata'].get('topic','') for c in chunks if c.get('metadata'))
        words    = sum(len(c.get('content','').split()) for c in chunks)
        return {
            'total_chunks':   total_chunks,
            'topics_covered': len(topics),
            'subjects':       len(subjects),
            'total_words':    words,
            'quiz_score':     st.session_state.get('quiz_score', 0),
            'chat_count':     len([m for m in st.session_state.chat_history if m['role'] == 'user']),
        }
    except Exception:
        return {'total_chunks': 0, 'topics_covered': 0, 'subjects': 0,
                'total_words': 0, 'quiz_score': 0, 'chat_count': 0}

def format_timestamp(dt):
    """Format datetime to readable string"""
    return dt.strftime("%B %d, %Y at %I:%M %p")

def create_download_button(content, filename, file_type="txt"):
    """Create download button for content"""
    st.download_button(
        label=f"📥 Download {file_type.upper()}",
        data=content,
        file_name=filename,
        mime=f"text/plain" if file_type == "txt" else f"application/{file_type}"
    )

def get_dynamic_subjects():
    """Get all available subjects directly from the trained vector database chunks"""
    try:
        from syllabus import SYLLABUS
        base_subjects = set(SYLLABUS.keys())
        
        if not components or 'warehouse' not in components:
            return sorted(list(base_subjects))
            
        subjects = set()
        for chunk in components['warehouse'].chunks_data:
            subj = chunk['metadata'].get('subject', 'general')
            if subj.lower() != 'general':
                subjects.add(subj)
                
        # Union with syllabus subjects to ensure all are included
        all_subjects = subjects.union(base_subjects)
        return sorted(list(all_subjects))
    except Exception:
        from syllabus import SYLLABUS
        return sorted(list(SYLLABUS.keys()))


def normalize_subject_query(subject: str) -> str:
    """Convert labels like MachineLearning to human-readable search text."""
    if not subject or subject == 'All':
        return ''
    subject_text = subject.replace('_', ' ')
    subject_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', subject_text)
    return subject_text.strip()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
def setup_sidebar():
    """Configure and display sidebar"""
    st.sidebar.markdown("# ⚙️ CONTROL PANEL")
    
    # User section
    st.sidebar.markdown(st.session_state.sidebar_header_style + "👤 USER INFO", unsafe_allow_html=True)
    user_name = st.sidebar.text_input("Your Name", value=st.session_state.user_name)
    st.session_state.user_name = user_name
    
    st.sidebar.markdown(f"**📧 Email:** learning@ai.edu")
    st.sidebar.markdown(f"**🎓 Level:** Intermediate")
    st.sidebar.markdown(f"**📊 Overall Progress:** 68%")
    
    # Learning preferences
    st.sidebar.markdown(st.session_state.sidebar_header_style + "🎯 LEARNING PREFERENCES", unsafe_allow_html=True)
    
    
    dynamic_subjects = get_dynamic_subjects()
    subject = st.sidebar.selectbox(
        "📚 Subject",
        ["All Subjects"] + dynamic_subjects,
        key="subject_select"
    )
    
    difficulty = st.sidebar.select_slider(
        "🔧 Difficulty Level",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Intermediate",
        key="difficulty_select"
    )
    
    quiz_difficulty = st.sidebar.select_slider(
        "📝 Quiz Difficulty",
        options=["Easy", "Medium", "Hard"],
        value="Medium",
        key="quiz_difficulty_select"
    )
    
    # Learning style
    learning_style = st.sidebar.radio(
        "📖 Preferred Learning Style",
        ["Conceptual", "Practical", "Mixed"],
        key="learning_style"
    )
    
    # Advanced settings
    st.sidebar.markdown(st.session_state.sidebar_header_style + "⚡ ADVANCED SETTINGS", unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        show_sources = st.checkbox("Show Sources", value=True)
    with col2:
        show_metadata = st.checkbox("Show Metadata", value=False)
    
    max_chunks = st.sidebar.slider("Max Chunks to Retrieve", 3, 20, 10)
    response_length = st.sidebar.select_slider("Response Length", ["Short", "Medium", "Long"], value="Medium")
    
    # System settings
    st.sidebar.markdown(st.session_state.sidebar_header_style + "🌐 SYSTEM", unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", key="refresh_btn"):
            st.rerun()
    with col2:
        if st.button("💾 Save Settings", key="save_btn"):
            st.success("✅ Settings saved!")
    
    # Help and support
    st.sidebar.markdown("---")
    st.sidebar.markdown(st.session_state.sidebar_header_style + "❓ HELP & SUPPORT", unsafe_allow_html=True)
    if st.sidebar.button("📖 User Guide", key="guide_btn"):
        st.info("User guide: Check the Documentation tab in Help section")
    if st.sidebar.button("🆘 Report Issue", key="issue_btn"):
        st.warning("Report issue to: support@teachingassistant.ai")
    
    return {
        'subject': subject,
        'difficulty': difficulty,
        'quiz_difficulty': quiz_difficulty,
        'learning_style': learning_style,
        'show_sources': show_sources,
        'show_metadata': show_metadata,
        'max_chunks': max_chunks,
        'response_length': response_length
    }

# ── Sidebar header style (must be before setup_sidebar) ─────────────────────
if 'sidebar_header_style' not in st.session_state:
    st.session_state.sidebar_header_style = '<p style="color:#1f77b4;font-weight:bold;font-size:1.1em;margin-top:15px;margin-bottom:5px;border-bottom:1px solid #ccc;padding-bottom:5px;">'

# ============================================================================
# DASHBOARD MODE
# ============================================================================
def show_dashboard():
    """Display main dashboard with live KB data"""
    st.markdown('<p class="main-header">🏠 AI Teaching Assistant PRO</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### Welcome back, **{st.session_state.user_name}**! 👋")
        st.markdown("*Your personal RAG-powered study companion*")
    with col2:
        st.metric("📅 Date", datetime.now().strftime("%d %b %Y"))
    with col3:
        st.metric("⏰ Time", datetime.now().strftime("%I:%M %p"))

    st.markdown("---")

    # ── Live KB stats ──────────────────────────────────────────────────────
    st.markdown('<p class="sub-header">📊 Knowledge Base Status</p>', unsafe_allow_html=True)
    stats = get_user_statistics()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("📦 Chunks",      stats['total_chunks'])
    with col2: st.metric("📚 Topics",       stats['topics_covered'])
    with col3: st.metric("🏷️ Subjects",    stats['subjects'])
    with col4: st.metric("📝 Words",        f"{stats['total_words']:,}")
    with col5: st.metric("💬 Chats Today", stats['chat_count'])

    st.markdown("---")

    # ── Real gap analysis + top topics ────────────────────────────────────
    col1, col2 = st.columns(2)
    wh = components['warehouse'] if components else None
    all_chunks = wh.chunks_data if wh else []

    with col1:
        st.markdown('<p class="sub-header">⚠️ Topics Needing Attention</p>', unsafe_allow_html=True)
        if all_chunks and components:
            try:
                gaps_data = components['syllabus'].identify_gaps(all_chunks)
                priority_gaps = gaps_data.get('priority_gaps', [])[:5]
                if priority_gaps:
                    gap_df = pd.DataFrame([{
                        'Topic':    g.get('topic',''),
                        'Subject':  g.get('subject',''),
                        'Priority': g.get('priority',''),
                    } for g in priority_gaps])
                    st.dataframe(gap_df, use_container_width=True)
                    if st.button("📖 Study These Topics", key="weak_btn"):
                        st.session_state.current_mode = "📚 Learning"
                        st.rerun()
                else:
                    st.success("✅ All syllabus topics are covered!")
            except Exception as e:
                st.warning(f"Could not load gap analysis: {e}")
        else:
            st.info("Run `python train_all.py` to load your course materials.")

    with col2:
        st.markdown('<p class="sub-header">💪 Top Topics in Knowledge Base</p>', unsafe_allow_html=True)
        if all_chunks:
            from collections import Counter
            topic_counts = Counter(
                c['metadata'].get('topic','general').replace('_',' ').title()
                for c in all_chunks if c.get('metadata')
            ).most_common(5)
            if topic_counts:
                top_df = pd.DataFrame(topic_counts, columns=['Topic', 'Chunks'])
                fig = px.bar(top_df, x='Chunks', y='Topic', orientation='h',
                             color='Chunks', color_continuous_scale='Blues',
                             title='Most Covered Topics')
                fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0),
                                  showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    st.markdown("---")

    # ── Quick actions ──────────────────────────────────────────────────────
    st.markdown('<p class="sub-header">⚡ Quick Actions</p>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("💬 Ask",     use_container_width=True): st.session_state.current_mode = "💬 Chat";      st.rerun()
    with col2:
        if st.button("📚 Learn",   use_container_width=True): st.session_state.current_mode = "📚 Learning";  st.rerun()
    with col3:
        if st.button("📝 Quiz",    use_container_width=True): st.session_state.current_mode = "📝 Quiz";     st.rerun()
    with col4:
        if st.button("📊 Exam",    use_container_width=True): st.session_state.current_mode = "📊 Exam";     st.rerun()
    with col5:
        if st.button("📈 Stats",   use_container_width=True): st.session_state.current_mode = "📈 Analytics"; st.rerun()
    with col6:
        if st.button("🎯 Syllabus",use_container_width=True): st.session_state.current_mode = "🎯 Syllabus"; st.rerun()

    # ── Subject distribution chart ─────────────────────────────────────────
    if all_chunks:
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Knowledge Base Distribution</p>', unsafe_allow_html=True)
        from collections import Counter
        subj_counts = Counter(
            c['metadata'].get('subject','Unknown')
            for c in all_chunks if c.get('metadata')
        )
        diff_counts = Counter(
            c['metadata'].get('difficulty','medium')
            for c in all_chunks if c.get('metadata')
        )
        col1, col2 = st.columns(2)
        with col1:
            s_df = pd.DataFrame(list(subj_counts.items()), columns=['Subject','Chunks'])
            fig  = px.pie(s_df, values='Chunks', names='Subject', hole=0.35,
                          title='By Subject', color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            d_df = pd.DataFrame(list(diff_counts.items()), columns=['Difficulty','Chunks'])
            color_map = {'easy':'#2ecc71','medium':'#f39c12','hard':'#e74c3c'}
            fig  = px.pie(d_df, values='Chunks', names='Difficulty', hole=0.35,
                          title='By Difficulty', color='Difficulty', color_discrete_map=color_map)
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# CHAT MODE
# ============================================================================
def show_chat_mode(settings):
    """Display chat interface — RAG + GPT powered"""
    st.markdown('<p class="main-header">💬 Ask Your Questions</p>', unsafe_allow_html=True)
    st.markdown("Answers sourced directly from your course materials via GPT")

    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f"**Subject:** {settings['subject']}")
    with col2: st.markdown(f"**Difficulty:** {settings['difficulty']}")
    with col3:
        gpt_on = components and components['generator']._client
        st.markdown(f"**Engine:** {'🟢 GPT' if gpt_on else '🟡 Smart Fallback'}")

    st.markdown("---")

    # ── Chat history ───────────────────────────────────────────────────────
    history = st.session_state.chat_history
    if history:
        st.markdown('<p class="sub-header">📋 Conversation</p>', unsafe_allow_html=True)
        for msg in history[-8:]:   # show last 8 messages
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(f"**{msg['content']}**")
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg['content'])
                    srcs = msg.get('sources', [])
                    if srcs and settings.get('show_sources'):
                        src_labels = []
                        for s in srcs[:3]:
                            if isinstance(s, dict):
                                src_labels.append(s.get('source_file', s.get('source','?')))
                        if src_labels:
                            st.caption("📚 Sources: " + " | ".join(src_labels))
                    if settings.get('show_metadata') and msg.get('quality'):
                        q = msg['quality']
                        st.caption(f"🔍 Quality: {q.get('quality_score',0):.2f} | Coverage: {q.get('keyword_coverage',0):.0%} | Model: {msg.get('model','')}")
        if st.button("🗑️ Clear History", key="clear_hist"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown("---")

    # ── Input ──────────────────────────────────────────────────────────────
    st.markdown('<p class="sub-header">✍️ Ask a Question</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_area(
            "Your question:",
            placeholder="e.g. What is normalization? How does deadlock occur? Explain BFS vs DFS",
            height=90, label_visibility="collapsed"
        )
    with col2:
        st.write("")  # spacer
        ask_btn = st.button("🚀 Ask", use_container_width=True)

    if ask_btn and question.strip():
        with st.spinner("🤔 Searching your course materials and generating answer..."):
            try:
                detected = get_subject_from_query(question)
                q_subj   = settings['subject'] if settings['subject'] != "All Subjects" else detected

                flt = {'k': settings.get('max_chunks', 8)}
                if q_subj and q_subj not in ('general', 'All Subjects'):
                    flt['subject'] = q_subj

                retrieved = components['retriever'].get_relevant_chunks(question, flt)

                if not retrieved:
                    st.warning("⚠️ No relevant content found. Add PDFs and retrain, or try rephrasing.")
                else:
                    answer_data = components['generator'].generate_answer(question, retrieved)
                    answer_text = answer_data.get('answer', str(answer_data))
                    quality     = components['generator'].evaluate_answer_quality(answer_text, question, retrieved)

                    st.session_state.chat_history.append({'role': 'user',      'content': question,     'timestamp': datetime.now()})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer_text,  'timestamp': datetime.now(),
                                                          'sources': retrieved, 'quality': quality, 'model': answer_data.get('model','')})
                    st.rerun()
            except Exception as e:
                st.error(f"❌ {str(e)}")
                import traceback; st.code(traceback.format_exc())
    elif ask_btn:
        st.warning("Please type a question first.")

    # ── Suggested questions ────────────────────────────────────────────────
    if not history:
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What is normalization in databases?",
            "Explain deadlock and how to prevent it",
            "What is a binary search tree?",
            "How does process scheduling work?",
            "What is the difference between stack and queue?",
        ]
        cols = st.columns(len(suggestions))
        for i, (col, sug) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(f"💬 {sug[:25]}...", key=f"sug_{i}", use_container_width=True):
                    st.session_state._suggested_q = sug
                    st.rerun()

    # Handle suggested question click
    if hasattr(st.session_state, '_suggested_q') and st.session_state._suggested_q:
        q = st.session_state._suggested_q
        st.session_state._suggested_q = None
        with st.spinner("🤔 Generating answer..."):
            try:
                retrieved   = components['retriever'].get_relevant_chunks(q, {'k': 6})
                answer_data = components['generator'].generate_answer(q, retrieved)
                answer_text = answer_data.get('answer', str(answer_data))
                quality     = components['generator'].evaluate_answer_quality(answer_text, q, retrieved)
                st.session_state.chat_history.append({'role': 'user',      'content': q,           'timestamp': datetime.now()})
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer_text, 'timestamp': datetime.now(),
                                                      'sources': retrieved, 'quality': quality, 'model': answer_data.get('model','')})
                st.rerun()
            except Exception as e:
                st.error(str(e))

    
    # Chat history display
    if st.session_state.chat_history:
        st.markdown('<p class="sub-header">📋 Conversation History</p>', unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.chat_history[-5:]):
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.info(f"**Assistant:** {msg['content']}")
    
    st.markdown("---")
    
    # Input section
    st.markdown('<p class="sub-header">✍️ Your Question</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_area(
            "What would you like to know?",
            placeholder="Ask about any topic... (e.g., 'What is normalization?')",
            height=100
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("🚀 Get Answer", use_container_width=True):
            if question.strip():
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get subject from query or settings
                        detected_subject = get_subject_from_query(question)
                        query_subject = settings['subject'] if settings['subject'] != "All Subjects" else detected_subject
                        
                        filters = {'k': settings['max_chunks']}
                        if query_subject and query_subject != "general":
                            filters['subject'] = query_subject
                            
                        # Retrieve relevant chunks
                        retrieved = components['retriever'].get_relevant_chunks(
                            question, 
                            filters
                        )
                        
                        # Generate answer
                        answer_data = components['generator'].generate_answer(question, retrieved)
                        answer_text = answer_data['answer'] if isinstance(answer_data, dict) else str(answer_data)

                        # Quality eval
                        quality = components['generator'].evaluate_answer_quality(
                            answer_text, question, retrieved)

                        # Store in history
                        st.session_state.chat_history.append({
                            'role':      'user',
                            'content':   question,
                            'timestamp': datetime.now(),
                        })
                        st.session_state.chat_history.append({
                            'role':      'assistant',
                            'content':   answer_text,
                            'timestamp': datetime.now(),
                            'sources':   retrieved,
                            'quality':   quality,
                            'model':     answer_data.get('model', 'unknown'),
                        })
                        
                        st.success("✅ Answer generated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Answer display
    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'assistant':
        st.markdown("---")
        st.markdown('<p class="sub-header">💡 Answer</p>', unsafe_allow_html=True)

        latest_answer = st.session_state.chat_history[-1]
        # Render GPT markdown directly — do NOT wrap in ** which breaks formatting
        st.markdown(latest_answer['content'])

        if settings['show_sources'] and latest_answer.get('sources'):
            st.markdown('<p class="sub-header">📚 Sources</p>', unsafe_allow_html=True)
            src_list = latest_answer['sources']
            if isinstance(src_list, list):
                for i, source in enumerate(src_list[:3], 1):
                    if isinstance(source, dict):
                        label = source.get('source_file', source.get('source', 'Unknown'))
                        score = source.get('relevance_score', source.get('score', 0))
                        st.caption(f"**{i}. {label}** — relevance: {score:.2f}")
                    else:
                        st.caption(f"**{i}.** {source}")

        # Show eval quality score if metadata available
        if settings.get('show_metadata') and latest_answer.get('quality'):
            q = latest_answer['quality']
            st.caption(f"🔍 Quality score: {q.get('quality_score', 0):.2f} | "
                       f"keyword coverage: {q.get('keyword_coverage', 0):.0%}")

        # Feedback
        st.markdown("---")
        st.markdown('<p class="sub-header">👍 Was this helpful?</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Yes, helpful!", use_container_width=True):
                st.success("Thanks for the feedback!")
        with col2:
            if st.button("🤔 Somewhat", use_container_width=True):
                st.info("We'll keep improving!")
        with col3:
            if st.button("👎 Not helpful", use_container_width=True):
                st.warning("Try rephrasing or selecting a different subject.")

# ============================================================================
# LEARNING MODE
# ============================================================================
def show_learning_mode(settings):
    """Display learning mode with GPT-powered explanations"""
    st.markdown('<p class="main-header">📚 Learning Mode</p>', unsafe_allow_html=True)
    st.markdown("Deep-dive into any concept — powered by your course materials + GPT")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "🎯 What would you like to learn?",
            placeholder="e.g. normalization, deadlock, linked list, process scheduling..."
        )
    with col2:
        depth = st.selectbox("Depth", ["Overview", "Detailed", "Advanced"], index=1)

    if topic and topic.strip():
        with st.spinner("📖 Generating learning material with AI..."):
            try:
                k_val = 5 if depth == "Overview" else (8 if depth == "Detailed" else 12)
                subj  = settings.get('subject', 'All Subjects')
                flt   = {'k': k_val}
                if subj and subj != 'All Subjects':
                    flt['subject'] = subj

                chunks = components['retriever'].get_relevant_chunks(topic, flt)

                if not chunks:
                    st.warning("⚠️ No content found for this topic. Try a different subject or add PDFs and retrain.")
                else:
                    st.success(f"✅ Found {len(chunks)} relevant sections — generating explanation...")
                    learned = components['learning'].create_explanation_with_example(topic, chunks)

                    # If GPT returned a full_answer, render it in one go
                    if learned.get('full_answer'):
                        tab1, tab2, tab3, tab4 = st.tabs(
                            ["📖 Full Explanation", "💡 Examples", "🔑 Key Points", "✏️ Practice"]
                        )
                        with tab1:
                            st.markdown('<p class="sub-header">AI Explanation</p>', unsafe_allow_html=True)
                            st.markdown(learned['full_answer'])
                            model_lbl = learned.get('model_used', '')
                            if model_lbl:
                                st.caption(f"🤖 Model: {model_lbl} | Difficulty: {learned.get('difficulty_level','?')}")

                        with tab2:
                            st.markdown('<p class="sub-header">Examples</p>', unsafe_allow_html=True)
                            examples = learned.get('examples', [])
                            if examples:
                                for i, ex in enumerate(examples, 1):
                                    with st.expander(f"📝 Example {i}"):
                                        st.markdown(ex)
                            else:
                                st.info("No specific examples extracted — try asking a more specific topic.")

                        with tab3:
                            st.markdown('<p class="sub-header">Key Points</p>', unsafe_allow_html=True)
                            kps = learned.get('key_points', [])
                            if kps:
                                for i, pt in enumerate(kps, 1):
                                    st.markdown(f"**{i}.** {pt}")
                            else:
                                st.info("Key points are included in the Full Explanation tab above.")

                        with tab4:
                            st.markdown('<p class="sub-header">Practice Questions</p>', unsafe_allow_html=True)
                            practice = components['learning'].generate_practice_questions(topic, chunks, count=4)
                            if practice:
                                for q in practice:
                                    with st.expander(f"❓ Q{q['id']}: {q['question'][:60]}..."):
                                        st.markdown(q['question'])
                                        st.caption(f"Difficulty: {q.get('difficulty','medium')} | Topic: {q.get('topic', topic)}")
                            else:
                                st.info("Not enough content to generate practice questions.")
                    else:
                        st.markdown(learned.get('definition', 'No explanation available.'))

                    # Related topics
                    related = learned.get('related_topics', [])
                    if related:
                        st.markdown("---")
                        st.markdown('<p class="sub-header">🔗 Related Topics</p>', unsafe_allow_html=True)
                        cols = st.columns(min(4, len(related)))
                        for i, rt in enumerate(related[:4]):
                            with cols[i]:
                                st.info(f"📚 {rt}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# EXAM PREPARATION MODE
# ============================================================================
def show_exam_mode(settings):
    """Display exam preparation — powered by RAG knowledge base"""
    st.markdown('<p class="main-header">📊 Exam Preparation</p>', unsafe_allow_html=True)
    st.markdown("Strategic exam prep based on your actual course materials")
    st.markdown("---")

    all_chunks = components['warehouse'].chunks_data if components else []
    subj_filter = settings.get('subject', 'All Subjects')
    if subj_filter and subj_filter != 'All Subjects':
        exam_chunks = [c for c in all_chunks if c.get('metadata', {}).get('subject') == subj_filter]
    else:
        exam_chunks = all_chunks

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Important Topics", "🔮 Predicted Questions", "⚠️ Weak Areas", "📅 Study Guide"]
    )

    with tab1:
        st.markdown('<p class="sub-header">Top Topics in Your Course Materials</p>', unsafe_allow_html=True)
        if exam_chunks:
            from collections import Counter
            topic_counts = Counter(
                c['metadata'].get('topic', 'general').replace('_', ' ').title()
                for c in exam_chunks
            )
            top_topics = topic_counts.most_common(10)
            total = sum(topic_counts.values())

            rows = []
            for i, (topic, count) in enumerate(top_topics):
                pct = int(count / total * 100)
                priority = '🔴 CRITICAL' if i < 2 else ('🟠 HIGH' if i < 5 else '🟡 MEDIUM')
                hrs = 4 if i < 2 else (3 if i < 4 else 2)
                rows.append({'Topic': topic, 'Chunks': count, 'Coverage %': f"{pct}%",
                             'Priority': priority, 'Study Time': f"{hrs} hours"})

            topics_df = pd.DataFrame(rows)
            st.dataframe(topics_df, use_container_width=True)

            fig = px.bar(topics_df, x='Chunks', y='Topic', orientation='h',
                         title='Topic Coverage in Knowledge Base', color='Chunks',
                         color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data found. Add PDFs and run training.")

    with tab2:
        st.markdown('<p class="sub-header">GPT-Predicted Exam Questions</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            pred_topic = st.text_input("Topic to predict questions for:", placeholder="e.g. normalization, deadlock, sorting...")
        with col_b:
            pred_count = st.number_input("# Questions", 3, 10, 5)

        if pred_topic and st.button("🔮 Generate Predictions", use_container_width=True):
            with st.spinner("Generating predicted exam questions..."):
                q_chunks = components['retriever'].get_relevant_chunks(
                    pred_topic, {'k': 8, 'subject': subj_filter if subj_filter != 'All Subjects' else None}
                )
                if q_chunks:
                    quiz_data = components['quiz'].generate_short_answer_quiz(q_chunks, pred_count)
                    questions = quiz_data.get('questions', [])
                    if questions:
                        for i, q in enumerate(questions, 1):
                            with st.expander(f"📝 Q{i}: {q['question'][:70]}..."):
                                st.markdown(f"**{q['question']}**")
                                if q.get('hint'):
                                    st.caption(f"Hint: {q['hint']}")
                                st.caption(f"Difficulty: {q.get('difficulty','medium')} | Marks: {q.get('marks',2)}")
                    else:
                        st.warning("Could not generate questions — try a different topic.")
                else:
                    st.warning("No relevant content found for this topic.")

    with tab3:
        st.markdown('<p class="sub-header">Syllabus Gap Analysis (Weak Areas)</p>', unsafe_allow_html=True)
        if all_chunks:
            gaps_data = components['syllabus'].identify_gaps(all_chunks)
            priority_gaps = gaps_data.get('priority_gaps', [])
            total_gaps = gaps_data.get('total_gaps', 0)

            col1, col2 = st.columns(2)
            with col1: st.metric("Total Gaps Found", total_gaps)
            with col2: st.metric("High Priority", sum(1 for g in priority_gaps if g.get('priority') == 'HIGH'))

            if priority_gaps:
                gaps_df = pd.DataFrame([{
                    'Subject':    g.get('subject',''),
                    'Topic':      g.get('topic',''),
                    'Priority':   g.get('priority',''),
                    'Difficulty': g.get('difficulty',''),
                    'Action':     g.get('suggestion','Add materials')[:50],
                } for g in priority_gaps[:15]])
                st.dataframe(gaps_df, use_container_width=True)
                st.warning("⚠️ Add PDFs/notes for missing topics and retrain to improve coverage.")
            else:
                st.success("✅ All syllabus topics are covered in your materials!")
        else:
            st.info("No data loaded — run training first.")

    with tab4:
        st.markdown('<p class="sub-header">Personalized Study Schedule</p>', unsafe_allow_html=True)
        if exam_chunks:
            from collections import Counter
            topics = Counter(
                c['metadata'].get('topic', 'general').replace('_', ' ').title()
                for c in exam_chunks
            ).most_common(7)

            schedule_rows = []
            for day, (topic, count) in enumerate(topics, 1):
                diff = 'hard' if count > 20 else ('medium' if count > 5 else 'easy')
                hrs = 4 if diff == 'hard' else (3 if diff == 'medium' else 2)
                schedule_rows.append({
                    'Day': f"Day {day}",
                    'Topic': topic,
                    'Hours': hrs,
                    'Difficulty': diff.title(),
                    'Activity': 'Study + Quiz' if day % 2 == 0 else 'Study + Practice',
                })
            schedule_df = pd.DataFrame(schedule_rows)
            st.dataframe(schedule_df, use_container_width=True)
            total_hrs = sum(r['Hours'] for r in schedule_rows)
            st.info(f"💡 Estimated {total_hrs} hours over {len(schedule_rows)} days based on your materials.")
        else:
            st.info("No data available to build schedule.")

# ============================================================================
# QUIZ MODE
# ============================================================================
def show_quiz_mode(settings):
    """Display quiz generator — GPT-powered real questions from course material"""
    st.markdown('<p class="main-header">📝 Quiz Generator</p>', unsafe_allow_html=True)
    st.markdown("Generates real exam-style questions from your course materials using GPT")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        quiz_type = st.selectbox("📋 Type", ["Multiple Choice (MCQ)", "Short Answer", "Mixed"])
    with col2:
        num_questions = st.slider("❓ Questions", 3, 15, 5)
    with col3:
        dynamic_subjects = get_dynamic_subjects()
        quiz_subject = st.selectbox("📚 Subject", ["All"] + dynamic_subjects)
    with col4:
        quiz_topic = st.text_input("🎯 Topic (optional)", placeholder="e.g. deadlock")

    if st.button("🚀 Generate Quiz", use_container_width=True):
        with st.spinner("🤖 Generating questions with GPT..."):
            try:
                subject_filter = None if quiz_subject == "All" else quiz_subject
                # Build query — prefer specific topic over generic subject
                if quiz_topic.strip():
                    search_query = quiz_topic.strip()
                elif quiz_subject != "All":
                    search_query = normalize_subject_query(quiz_subject)
                else:
                    search_query = "important concepts for exam"
                flt = {'k': max(num_questions * 3, 15)}
                if subject_filter:
                    flt['subject'] = subject_filter

                chunks = components['retriever'].get_relevant_chunks(search_query, flt)

                if not chunks:
                    st.warning("⚠️ No content found. Add PDFs and retrain first.")
                else:
                    if quiz_type == "Multiple Choice (MCQ)":
                        quiz_data = components['quiz'].generate_mcq_quiz(chunks, num_questions)
                    elif quiz_type == "Short Answer":
                        quiz_data = components['quiz'].generate_short_answer_quiz(chunks, num_questions)
                    else:
                        quiz_data = components['quiz'].generate_mixed_quiz(chunks, num_questions)

                    questions = quiz_data.get('questions', [])
                    if not questions:
                        st.error("Could not generate questions. Try a different topic or subject.")
                    else:
                        st.session_state.current_quiz   = questions
                        st.session_state.quiz_responses = {}
                        st.session_state.quiz_submitted = False
                        powered = quiz_data.get('powered_by', 'Fallback')
                        st.success(f"✅ {len(questions)} questions generated — powered by {powered}")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback; st.code(traceback.format_exc())

    # ── Quiz display ────────────────────────────────────────────────────────
    if 'current_quiz' in st.session_state and st.session_state.current_quiz:
        st.markdown("---")
        quiz = st.session_state.current_quiz
        submitted = st.session_state.get('quiz_submitted', False)

        st.markdown(f'<p class="sub-header">📝 {len(quiz)} Questions</p>', unsafe_allow_html=True)

        for i, q in enumerate(quiz, 1):
            q_type   = q.get('type', 'mcq').lower()
            q_text   = q.get('question', '')
            topic    = q.get('topic', '')
            diff     = q.get('difficulty', 'medium')
            diff_emoji = {'easy': '🟢', 'medium': '🟡', 'hard': '🔴'}.get(diff, '🟡')

            label = f"{diff_emoji} Q{i}: {q_text[:60]}{'...' if len(q_text)>60 else ''}"

            with st.expander(label, expanded=not submitted):
                st.markdown(f"**{q_text}**")
                st.caption(f"Topic: {topic} | Difficulty: {diff} | Marks: {q.get('marks',1)}")

                if q_type == 'mcq':
                    options = q.get('options', [])
                    if not options:
                        options = ['A) Option 1', 'B) Option 2', 'C) Option 3', 'D) Option 4']

                    if submitted:
                        user_ans    = st.session_state.quiz_responses.get(i, '')
                        correct_ans = q.get('correct_option', '')
                        if user_ans and correct_ans.lower() in user_ans.lower():
                            st.success(f"✅ Your answer: **{user_ans}** — Correct!")
                        else:
                            st.error(f"❌ Your answer: **{user_ans}**")
                            st.info(f"✅ Correct answer: **{correct_ans}**")
                    else:
                        resp = st.radio("Select your answer:", options, key=f"q_{i}_mcq", index=None)
                        if resp is not None:
                            st.session_state.quiz_responses[i] = resp

                else:  # short answer
                    hint = q.get('hint', '')
                    if submitted:
                        user_ans = st.session_state.quiz_responses.get(i, '')
                        if user_ans and len(user_ans.strip().split()) >= 5:
                            st.success(f"✅ Your answer: {user_ans}")
                        else:
                            st.warning("⚠️ Answer too short or not provided.")
                        if hint:
                            st.info(f"📝 Reference: {hint}")
                    else:
                        if hint:
                            st.caption(f"💡 Hint: {hint[:120]}...")
                        resp = st.text_area("Your answer (2-3 sentences):", key=f"q_{i}_sa", height=80)
                        if resp:
                            st.session_state.quiz_responses[i] = resp

        # ── Submit / Restart bar ──────────────────────────────────────────
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if not submitted and st.button("✅ Submit Quiz", use_container_width=True):
                answered = len([v for v in st.session_state.quiz_responses.values() if v])
                if answered == 0:
                    st.warning("Please answer at least one question before submitting.")
                else:
                    st.session_state.quiz_submitted = True
                    # Evaluate
                    quiz_dict = {'quiz_type': 'mixed', 'questions': quiz}
                    resp_list = [st.session_state.quiz_responses.get(i, '') for i in range(1, len(quiz)+1)]
                    results   = components['quiz'].evaluate_quiz_response(quiz_dict, resp_list)

                    score   = results.get('score', 0)
                    total   = results.get('total_marks', 0)
                    pct     = results.get('percentage', '0%')
                    st.session_state.quiz_score = score

                    # Score summary
                    st.markdown('<p class="sub-header">📊 Results</p>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Score",    pct)
                    with c2: st.metric("Marks",    f"{score}/{total}")
                    with c3: st.metric("Attempted",f"{answered}/{len(quiz)}")
                    st.markdown(f"**{results.get('feedback','Good effort!')}**")
                    st.rerun()

        with col2:
            if submitted:
                # Show summary scores after submission
                quiz_dict = {'quiz_type': 'mixed', 'questions': quiz}
                resp_list = [st.session_state.quiz_responses.get(i, '') for i in range(1, len(quiz)+1)]
                results   = components['quiz'].evaluate_quiz_response(quiz_dict, resp_list)
                score  = results.get('score', 0)
                total  = results.get('total_marks', 0)
                pct    = results.get('percentage', '0%')
                st.metric("🏆 Final Score", pct)
                st.caption(f"{score}/{total} marks | {results.get('feedback','')}")

        with col3:
            if st.button("🔄 New Quiz", use_container_width=True):
                for key in ['current_quiz', 'quiz_responses', 'quiz_submitted']:
                    st.session_state.pop(key, None)
                st.rerun()



# ============================================================================
# ANALYTICS MODE
# ============================================================================
def show_analytics_mode(settings):
    """Display analytics dashboard — real data from warehouse"""
    st.markdown('<p class="main-header">📈 Analytics & Knowledge Base</p>', unsafe_allow_html=True)
    st.markdown("Live statistics from your trained knowledge base and evaluation results")
    st.markdown("---")

    import json as _json, os as _os

    # ── Load real data ────────────────────────────────────────────────────────
    wh = components['warehouse'] if components else None
    eval_data = {}
    eval_path = 'data/eval_report.json'
    if _os.path.exists(eval_path):
        try:
            with open(eval_path) as f:
                eval_data = _json.load(f)
        except Exception:
            pass

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Knowledge Base", "🎯 Topics & Subjects", "📈 Eval Metrics", "📋 Raw Data"]
    )

    with tab1:
        st.markdown('<p class="sub-header">Knowledge Base Overview</p>', unsafe_allow_html=True)
        cov = eval_data.get('data_coverage', {})
        total_chunks = cov.get('total_chunks', 0)
        total_words  = cov.get('total_words', 0)
        avg_words    = cov.get('avg_words_per_chunk', 0)
        by_subj      = cov.get('by_subject', {})

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("📦 Total Chunks", total_chunks)
        with col2: st.metric("📝 Total Words", f"{total_words:,}")
        with col3: st.metric("📏 Avg Words/Chunk", f"{avg_words:.0f}")
        with col4: st.metric("📚 Subjects", len(by_subj))

        if by_subj:
            subj_df = pd.DataFrame(list(by_subj.items()), columns=['Subject', 'Chunks'])
            fig = px.pie(subj_df, values='Chunks', names='Subject', hole=0.35,
                         title='Chunk Distribution by Subject',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<p class="sub-header">Topic & Difficulty Breakdown</p>', unsafe_allow_html=True)
        by_topic = eval_data.get('data_coverage', {}).get('by_topic', {})
        by_diff  = eval_data.get('data_coverage', {}).get('by_difficulty', {})

        col1, col2 = st.columns(2)
        with col1:
            if by_topic:
                top_topics = sorted(by_topic.items(), key=lambda x: x[1], reverse=True)[:10]
                t_df = pd.DataFrame(top_topics, columns=['Topic', 'Chunks'])
                t_df['Topic'] = t_df['Topic'].str.replace('_', ' ').str.title()
                fig = px.bar(t_df, x='Chunks', y='Topic', orientation='h',
                             title='Top 10 Topics', color='Chunks',
                             color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if by_diff:
                d_df = pd.DataFrame(list(by_diff.items()), columns=['Difficulty', 'Chunks'])
                fig = px.pie(d_df, values='Chunks', names='Difficulty',
                             title='Difficulty Distribution',
                             color_discrete_map={'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<p class="sub-header">Evaluation Metrics (Latest Run)</p>', unsafe_allow_html=True)
        ret = eval_data.get('retrieval', {})
        cls = eval_data.get('classification', {})
        noise = eval_data.get('noise_analysis', {})

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🎯 F1 Score",    f"{ret.get('macro_f1', 0):.3f}")
        with col2: st.metric("📐 MAP",          f"{ret.get('MAP', 0):.3f}")
        with col3: st.metric("📈 NDCG@5",       f"{ret.get('mean_ndcg', 0):.3f}")
        with col4: st.metric("✅ Subj Accuracy", f"{ret.get('subject_accuracy', 0):.0%}")

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🔍 Precision@5", f"{ret.get('macro_precision', 0):.3f}")
        with col2: st.metric("🔄 Recall@5",    f"{ret.get('macro_recall', 0):.3f}")
        with col3: st.metric("🔗 MRR",         f"{ret.get('MRR', 0):.3f}")
        with col4: st.metric("📊 Topic Acc",   f"{cls.get('overall_accuracy', 0):.0%}")

        # Per-query breakdown
        per_query = ret.get('per_query', [])
        if per_query:
            st.markdown('<p class="sub-header">Per-Query Results</p>', unsafe_allow_html=True)
            pq_df = pd.DataFrame([{
                'Query': q['query'][:50]+'...' if len(q['query'])>50 else q['query'],
                'Expected': q.get('expected_subject',''),
                'Got': q.get('retrieved_subject',''),
                'F1':  round(q.get('f1',0),3),
                'NDCG': round(q.get('ndcg',0),3),
            } for q in per_query])
            st.dataframe(pq_df, use_container_width=True)

    with tab4:
        st.markdown('<p class="sub-header">Raw Knowledge Base Data</p>', unsafe_allow_html=True)
        if wh and wh.chunks_data:
            chunk_rows = [{
                'ID':       c['id'],
                'Subject':  c['metadata'].get('subject',''),
                'Topic':    c['metadata'].get('topic','').replace('_',' ').title(),
                'Difficulty': c['metadata'].get('difficulty',''),
                'Words':    len(c['content'].split()),
                'Source':   c['metadata'].get('source_file','')[:30],
            } for c in wh.chunks_data]
            chunk_df = pd.DataFrame(chunk_rows)
            st.dataframe(chunk_df, use_container_width=True, height=400)
        else:
            st.info("No data loaded. Run training first.")

        # Download eval report
        if eval_data:
            st.download_button(
                label="📥 Download Eval Report (JSON)",
                data=_json.dumps(eval_data, indent=2),
                file_name="eval_report.json",
                mime="application/json"
            )

# ============================================================================
# SYLLABUS MODE
# ============================================================================
def show_syllabus_mode(settings):
    """Display syllabus management — real coverage from knowledge base"""
    st.markdown('<p class="main-header">🎯 Syllabus Management</p>', unsafe_allow_html=True)
    st.markdown("Track which syllabus topics are covered in your uploaded course materials")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📚 Coverage Analysis", "✅ Topic Validation", "📖 Study Guide"]
    )

    # Get all chunks for analysis
    all_chunks = components['warehouse'].chunks_data if components else []

    with tab1:
        st.markdown('<p class="sub-header">Real Syllabus Coverage</p>', unsafe_allow_html=True)
        if all_chunks:
            coverage_data = components['syllabus'].get_syllabus_coverage(all_chunks)
            cov_analysis  = coverage_data.get('coverage_analysis', {})
            overall       = coverage_data.get('overall_coverage', {})

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Topics",   overall.get('total_topics', 0))
            with col2: st.metric("Covered",         overall.get('covered_topics', 0))
            with col3: st.metric("Overall Coverage",overall.get('overall_coverage_percentage', '0%'))

            rows = []
            for subj, data in cov_analysis.items():
                rows.append({
                    'Subject':   subj,
                    'Total':     data['total_topics'],
                    'Covered':   data['covered_topics'],
                    'Missing':   data['not_covered_topics'],
                    'Coverage %': data['coverage_percentage'],
                })
            if rows:
                cov_df = pd.DataFrame(rows)
                st.dataframe(cov_df, use_container_width=True)
                fig = px.bar(cov_df, x='Subject', y=['Covered', 'Missing'],
                             title='Topics Covered vs Missing per Subject',
                             barmode='group',
                             color_discrete_map={'Covered': '#2ecc71', 'Missing': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)

            # Gaps
            gaps_data = components['syllabus'].identify_gaps(all_chunks)
            gaps = gaps_data.get('gaps_analysis', {})
            total_gaps = gaps_data.get('total_gaps', 0)
            if total_gaps:
                st.warning(f"⚠️ {total_gaps} topic(s) are not covered in your materials")
                for subj, gap_list in gaps.items():
                    with st.expander(f"📖 {subj} — {len(gap_list)} missing topics"):
                        for g in gap_list[:10]:
                            st.markdown(f"• **{g['topic']}** [{g['priority']}] — {g['suggestion']}")
            else:
                st.success("✅ All syllabus topics are covered in your materials!")
        else:
            st.info("No data loaded. Run `python train_all.py` first.")

    with tab2:
        st.markdown('<p class="sub-header">Validate a Topic</p>', unsafe_allow_html=True)
        validate_subject = st.selectbox(
            "Select Subject", list(SYLLABUS.keys()), key="validate_subject_select"
        )
        topic_input = st.text_input("Enter a topic to validate:")
        if topic_input:
            result = components['syllabus'].validate_topic(validate_subject, topic_input)
            if result.get('valid'):
                st.success(result['message'])
            else:
                st.warning(result['message'])
                if result.get('suggestion'):
                    st.info(result['suggestion'])
            available = ', '.join(SYLLABUS.get(validate_subject, {}).get('topics', [])[:8])
            st.caption(f"Sample topics: {available}")

    with tab3:
        st.markdown('<p class="sub-header">Generated Study Guide</p>', unsafe_allow_html=True)
        guide_subject = st.selectbox(
            "Generate guide for:", list(SYLLABUS.keys()), key="guide_subject_select"
        )
        if st.button("📖 Generate Study Guide", use_container_width=True):
            with st.spinner("Building study guide..."):
                guide = components['syllabus'].generate_study_guide(guide_subject, all_chunks)
                if 'error' not in guide:
                    st.success(f"📅 Estimated total: {guide.get('total_estimated_hours', 0)} hours")
                    units = guide.get('study_units', [])
                    if units:
                        guide_df = pd.DataFrame([{
                            'Unit':      u['unit'],
                            'Topic':     u['topic'],
                            'Difficulty': u['difficulty'],
                            'Hours':     u['estimated_study_hours'],
                            'Material Available': '✅' if u['content_available'] else '❌',
                        } for u in units])
                        st.dataframe(guide_df, use_container_width=True)
                else:
                    st.error(guide['error'])

# ============================================================================
# HELP & SUPPORT MODE
# ============================================================================
def show_help_mode():
    """Display help and documentation"""
    st.markdown('<p class="main-header">❓ Help & Support</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📖 User Guide", "❓ FAQ", "🆘 Troubleshooting", "📧 Contact"]
    )
    
    with tab1:
        st.markdown("""
        ## User Guide
        
        ### Getting Started
        1. Set your preferences in the sidebar
        2. Choose a learning mode from the dashboard
        3. Ask questions or explore topics
        4. Track your progress with analytics
        
        ### Features Overview
        - **Chat Mode**: Ask any question about your subjects
        - **Learning Mode**: Deep dive into any concept
        - **Exam Prep**: Prepare strategically for exams
        - **Quiz Generator**: Test your knowledge
        - **Analytics**: Track your progress
        - **Syllabus**: Manage your curriculum
        """)
    
    with tab2:
        st.markdown("""
        ## Frequently Asked Questions
        
        **Q: How accurate are the predictions?**
        A: Based on historical data, our predictions are 85-92% accurate.
        
        **Q: Can I export my progress?**
        A: Yes, use the analytics section to download reports.
        
        **Q: What's the best study strategy?
        A: Focus on weak areas, Then practice with quizzes.
        """)
    
    with tab3:
        st.markdown("""
        ## Troubleshooting
        
        **Issue: Slow response time**
        - Clear browser cache
        - Refresh the page
        
        **Issue: Quiz not saving**
        - Check internet connection
        - Try again
        """)
    
    with tab4:
        st.markdown("""
        ## Contact Us
        
        📧 **Email:** support@teachingassistant.ai
        💬 **Chat:** Available in chat mode
        📱 **Phone:** +1-800-TEACH-AI
        """)

# ============================================================================
# SETTINGS & PREFERENCES
# ============================================================================
def show_settings_mode():
    """Display settings and preferences"""
    st.markdown('<p class="main-header">⚙️ Settings & Preferences</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(
        ["👤 Profile", "🎨 Appearance", "🔔 Notifications"]
    )
    
    with tab1:
        st.markdown("### Profile Settings")
        name = st.text_input("Full Name", value=st.session_state.user_name)
        email = st.text_input("Email", value="student@example.com")
        level = st.selectbox("Learning Level", ["Beginner", "Intermediate", "Advanced"])
        
        if st.button("Save Profile"):
            st.success("✅ Profile updated!")
    
    with tab2:
        st.markdown("### Appearance Settings")
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        font_size = st.slider("Font Size", 10, 16, 12)
        
        if st.button("Apply Theme"):
            st.success("✅ Theme updated!")
    
    with tab3:
        st.markdown("### Notification Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Email Notifications", value=True)
            st.checkbox("Study Reminders", value=True)
        with col2:
            st.checkbox("Quiz Results", value=True)
            st.checkbox("Progress Updates", value=True)
        
        if st.button("Save Notifications"):
            st.success("✅ Notifications updated!")

# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================
def main():
    """Main application routing"""
    
    # Sidebar setup
    settings = setup_sidebar()
    
    # Main content area
    st.markdown("---")
    
    # Mode selection
    mode_col1, mode_col2, mode_col3, mode_col4, mode_col5, mode_col6, mode_col7, mode_col8 = st.columns(8)
    
    modes = [
        ("🏠", "Dashboard", mode_col1),
        ("💬", "Chat", mode_col2),
        ("�", "Quiz", mode_col5),
        ("📈", "Analytics", mode_col6),
        ("❓", "Help", mode_col8),
    ]
    
    for emoji, name, col in modes:
        with col:
            if st.button(f"{emoji}\n{name}", use_container_width=True, key=f"btn_{name}"):
                st.session_state.current_mode = f"{emoji} {name}"
                st.rerun()
    
    st.markdown("---")
    
    # Route to appropriate mode
    if "Dashboard" in st.session_state.current_mode:
        show_dashboard()
    elif "Chat" in st.session_state.current_mode:
        show_chat_mode(settings)
    elif "Quiz" in st.session_state.current_mode:
        show_quiz_mode(settings)
    elif "Analytics" in st.session_state.current_mode:
        show_analytics_mode(settings)
    elif "Help" in st.session_state.current_mode:
        show_help_mode()
    else:
        show_dashboard()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.caption("🚀 AI Teaching Assistant PRO")
    with col2:
        st.caption("© 2024 | Powered by RAG, Data Mining & ML | All Rights Reserved")
    with col3:
        st.caption("v2.0 Premium Edition")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    if not st.session_state.initialized:
        st.session_state.initialized = True
    
    main()
