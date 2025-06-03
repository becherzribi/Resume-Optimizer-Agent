import hashlib
import streamlit as st
import sqlite3
import os
import io
import re
import logging # Keep standard logging for basic setup
import structlog # For structured logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from passlib.context import CryptContext
# ReportLab imports are now primarily in utils.py for PDF generation functions

# --- Local Imports ---
from parser import extract_text_from_pdf
from llm_engine import analyze_resume, generate_enhanced_resume, get_model_recommendations, llm_engine
from utils import (
    compute_token_similarity,
    find_missing_keywords,
    check_date_formatting,
    advanced_date_formatting_check,
    highlight_missing_keywords,
    suggest_action_verbs,
    calculate_keyword_density,
    generate_professional_pdf_bytes, # This is now the primary PDF generator from utils
    text_to_pdf_bytes_basic # This is the fallback PDF generator from utils
)
from semantic_search import SemanticMatcher, create_semantic_matcher

# --- Configuration & Initialization ---
load_dotenv()

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
log = structlog.get_logger(__name__) # Use __name__ for better context in logs

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(name)s - %(levelname)s - %(message)s",
)

# --- Constants & Environment Variables ---
DATABASE_NAME = os.getenv("DATABASE_NAME", "resume_optimizer.db")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache/semantic_indexes"))
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1") # Example model
PROMPT_DIR = Path("prompts")
ENHANCED_PROMPT_FILE = PROMPT_DIR / "enhanced_resume_review.txt"
LOGO_PATH = "logo.png"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'authenticated': False,
        'username': None,
        'analysis_results': None,
        'cached_job_semantic_matcher': None,
        'cached_job_hash_for_matcher': None,
        'confirm_delete_history': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Database Functions ---
def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE_NAME, timeout=10) # Added timeout
        conn.row_factory = sqlite3.Row
        log.debug("Database connection established", db_name=DATABASE_NAME)
        return conn
    except sqlite3.Error as e:
        log.error("Database connection failed", error=str(e), db_name=DATABASE_NAME, exc_info=True)
        st.error("Database connection error. Please try again or contact support.")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            with conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS users (
                                username TEXT PRIMARY KEY,
                                password_hash TEXT NOT NULL
                             )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS analyses (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                username TEXT NOT NULL,
                                resume_name TEXT,
                                job_name TEXT,
                                analysis_date TEXT NOT NULL,
                                feedback TEXT,
                                similarity_score REAL,
                                missing_keywords_count INTEGER,
                                date_issues_count INTEGER,
                                model_used TEXT,
                                target_role TEXT,
                                language TEXT,
                                FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
                             )""") # Added ON DELETE CASCADE
            log.info("Database schema initialized/verified.")
        except sqlite3.Error as e:
            log.error("Database initialization failed", error=str(e), exc_info=True)
        finally:
            conn.close()

def delete_user_analyses(username):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn:
            conn.execute("DELETE FROM analyses WHERE username = ?", (username,))
        log.info("User analyses deleted", username=username)
        return True
    except sqlite3.Error as e:
        log.error("Failed to delete user analyses", error=str(e), username=username, exc_info=True)
        st.error(f"Failed to delete analysis history: {e}")
        return False
    finally:
        conn.close()

# --- Authentication Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

def authenticate_user(username, password):
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user_record = cursor.fetchone()
        if user_record and verify_password(password, user_record['password_hash']):
            log.info("User authentication successful", username=username)
            return True
        log.warning("User authentication failed", username=username, reason="Invalid credentials" if user_record else "User not found")
        return False
    except sqlite3.Error as e:
        log.error("Authentication database error", error=str(e), username=username, exc_info=True)
        return False
    finally:
        conn.close()

def register_user(username, password):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn:
            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                         (username, hash_password(password)))
        log.info("User registration successful", username=username)
        return True
    except sqlite3.IntegrityError:
        log.warning("User registration failed: Username already exists", username=username)
        return False
    except sqlite3.Error as e:
        log.error("User registration database error", error=str(e), username=username, exc_info=True)
        return False
    finally:
        conn.close()

# --- Analysis Data Functions ---
def save_analysis(username, analysis_data_dict):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn:
            feedback_content_to_save = analysis_data_dict.get('llm_feedback_to_save', analysis_data_dict.get('llm_feedback', ''))
            conn.execute("""INSERT INTO analyses (username, resume_name, job_name, analysis_date,
                                             feedback, similarity_score, missing_keywords_count,
                                             date_issues_count, model_used, target_role, language)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (username,
                          analysis_data_dict.get('resume_name', 'N/A'),
                          analysis_data_dict.get('job_name', 'N/A'),
                          datetime.now().isoformat(),
                          feedback_content_to_save,
                          analysis_data_dict.get('token_sim_score', 0.0),
                          len(analysis_data_dict.get('missing_keywords', [])),
                          len(analysis_data_dict.get('basic_date_issues', [])) + len(analysis_data_dict.get('advanced_date_issues', [])),
                          analysis_data_dict.get('model_choice', 'N/A'),
                          analysis_data_dict.get('target_role', ''),
                          analysis_data_dict.get('language', 'N/A')))
        log.info("Analysis results saved", username=username, resume_name=analysis_data_dict.get('resume_name'))
        return True
    except sqlite3.Error as e:
        log.error("Failed to save analysis results", error=str(e), username=username, exc_info=True)
        st.error(f"Database error while saving analysis: {e}")
        return False
    finally:
        conn.close()

def get_user_analyses(username):
    conn = get_db_connection()
    if not conn: return []
    try:
        cursor = conn.execute("""SELECT id, resume_name, job_name, analysis_date, feedback, model_used, target_role, language
                                 FROM analyses WHERE username = ? ORDER BY analysis_date DESC""", (username,))
        analyses_rows = cursor.fetchall()
        log.debug("Retrieved past analyses for user", username=username, count=len(analyses_rows))
        return [dict(row) for row in analyses_rows] # Convert rows to dicts
    except sqlite3.Error as e:
        log.error("Failed to retrieve past analyses", error=str(e), username=username, exc_info=True)
        return []
    finally:
        conn.close()

# --- Core Analysis Logic ---
@st.cache_data(show_spinner=False, ttl=3600) # Added TTL for cache
def run_full_analysis(_resume_bytes_tuple, _job_bytes_tuple, resume_name, job_name, model_choice, target_role, language):
    # Unpack tuples passed by st.cache_data for byte-like objects
    _resume_bytes, _job_bytes = _resume_bytes_tuple[0], _job_bytes_tuple[0]

    analysis_results = {}
    analysis_results['resume_name'] = resume_name
    analysis_results['job_name'] = job_name
    analysis_results['model_choice'] = model_choice
    analysis_results['target_role'] = target_role
    analysis_results['language'] = language

    progress_bar_area = st.empty() # For both text and bar

    total_steps = 7

    def update_progress(step_num, text_message):
        with progress_bar_area.container():
            st.caption(f"⏳ {text_message}")
            st.progress((step_num + 1) / total_steps) # step_num is 0-indexed

    try:
        update_progress(0, "Initializing analysis...")

        update_progress(1, "Extracting text from resume...")
        resume_text = extract_text_from_pdf(io.BytesIO(_resume_bytes))
        analysis_results['resume_text'] = resume_text
        log.info("Resume text extracted", length=len(resume_text) if resume_text else 0)

        update_progress(2, "Processing job description...")
        job_file_obj = io.BytesIO(_job_bytes)
        if job_name.lower().endswith('.txt'):
            job_text = job_file_obj.read().decode("utf-8", errors="replace")
        else:
            job_text = extract_text_from_pdf(job_file_obj)
        analysis_results['job_text'] = job_text
        log.info("Job description text extracted", length=len(job_text) if job_text else 0)

        update_progress(3, "Building semantic understanding model...")
        job_hash = hashlib.md5(job_text.encode("utf-8")).hexdigest()
        sem_matcher_instance = None
        if 'cached_job_semantic_matcher' not in st.session_state or \
           st.session_state.get('cached_job_hash_for_matcher') != job_hash or \
           st.session_state.cached_job_semantic_matcher.language != language:
            log.info("Semantic cache miss/invalid. Building new job index.", job_hash=job_hash, language=language)
            job_sentences_list = [line.strip() for line in job_text.splitlines() if len(line.strip()) > 15] # Slightly longer sentences
            if not job_sentences_list:
                st.warning("Could not extract enough significant sentences from job description for semantic analysis.")
            else:
                sem_matcher_instance = create_semantic_matcher(language=language)
                sem_matcher_instance.build_index(job_sentences_list)
                st.session_state.cached_job_semantic_matcher = sem_matcher_instance
                st.session_state.cached_job_hash_for_matcher = job_hash
        else:
            log.info("Using cached semantic index for job description.", job_hash=job_hash)
            sem_matcher_instance = st.session_state.cached_job_semantic_matcher
        analysis_results['semantic_matcher'] = sem_matcher_instance

        update_progress(4, "Performing keyword & formatting checks...")
        analysis_results['token_sim_score'] = compute_token_similarity(resume_text, job_text)
        analysis_results['missing_keywords'] = find_missing_keywords(resume_text, job_text)
        analysis_results['basic_date_issues'] = check_date_formatting(resume_text)
        analysis_results['advanced_date_issues'] = advanced_date_formatting_check(resume_text)
        log.info("Keyword and formatting checks complete", score=analysis_results['token_sim_score'])

        update_progress(5, "Finding semantic links...")
        if sem_matcher_instance:
            resume_sentences_list = [line.strip() for line in resume_text.splitlines() if len(line.strip()) > 15]
            if resume_sentences_list:
                analysis_results['semantic_results'] = sem_matcher_instance.query(resume_sentences_list, top_k=3)
                log.info("Semantic matching complete.")
            else:
                analysis_results['semantic_results'] = {}
                st.warning("Could not extract enough significant sentences from resume for semantic analysis.")
        else:
             analysis_results['semantic_results'] = {}

        update_progress(6, f"Generating AI feedback with {model_choice}...")
        if not ENHANCED_PROMPT_FILE.exists():
            error_msg = f"Critical Error: Prompt file not found ({ENHANCED_PROMPT_FILE})"
            st.error(error_msg)
            log.critical("Prompt file missing", path=str(ENHANCED_PROMPT_FILE))
            analysis_results['llm_feedback'] = error_msg
        else:
            with open(ENHANCED_PROMPT_FILE, "r", encoding="utf-8") as f:
                prompt_template_content = f.read()
            analysis_results['llm_feedback'] = analyze_resume(
                resume_text=resume_text,
                job_text=job_text,
                prompt_template_string=prompt_template_content,
                model_name=model_choice,
                target_role=target_role,
                language=language
            )
            log.info("LLM feedback generated", model=model_choice)

        update_progress(7, "Analysis finalized!")
        return analysis_results

    except Exception as e:
        log.error("Core analysis pipeline encountered an error", error=str(e), exc_info=True)
        st.error(f"An unexpected error occurred during analysis: {e}")
        progress_bar_area.empty()
        return None
    finally:
        progress_bar_area.empty()


# --- Streamlit UI Functions ---
def display_login_register():
    st.set_page_config(page_title="Resume Optimizer Agent", layout="wide", initial_sidebar_state="collapsed")
    if Path(LOGO_PATH).is_file():
      st.image(LOGO_PATH, width=120)
    else:
      st.markdown("## 🧾 Resume Optimizer Agent")


    st.markdown("#### Welcome! Let's optimize your resume against job descriptions using AI.")
    st.divider()

    login_tab, register_tab = st.tabs(["🔐 **Login**", "✨ **Create Account**"])

    with login_tab:
        with st.form("login_form_main_page"):
            username_login = st.text_input("Username", key="login_username_field", placeholder="Your username")
            password_login = st.text_input("Password", type="password", key="login_password_field", placeholder="Your password")
            login_submitted_btn = st.form_submit_button("Login", type="primary", use_container_width=True)
            if login_submitted_btn:
                if authenticate_user(username_login, password_login):
                    st.session_state.authenticated = True
                    st.session_state.username = username_login
                    log.info("User logged in", username=username_login)
                    st.success("Logged in successfully! Redirecting...")
                    st.balloons()
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password. Please try again or register.")

    with register_tab:
        with st.form("register_form_main_page"):
            new_username_reg = st.text_input("Choose a Username", key="reg_username_field", placeholder="Min. 4 characters")
            new_password_reg = st.text_input("Choose a Password", type="password", key="reg_password_field", placeholder="Min. 8 characters, include numbers/symbols")
            confirm_password_reg = st.text_input("Confirm Password", type="password", key="reg_confirm_password_field")
            register_submitted_btn = st.form_submit_button("Register & Auto-Login", use_container_width=True)

            if register_submitted_btn:
                if not new_username_reg or not new_password_reg:
                    st.error("Username and password fields cannot be empty.")
                elif len(new_username_reg) < 4:
                    st.error("Username must be at least 4 characters long.")
                elif new_password_reg != confirm_password_reg:
                    st.error("Passwords do not match!")
                elif len(new_password_reg) < 8: # Add more complex regex for password strength if needed
                    st.error("Password must be at least 8 characters long.")
                else:
                    if register_user(new_username_reg, new_password_reg):
                        st.success(f"Welcome, {new_username_reg}! Registration successful. You are now logged in.")
                        log.info("New user registered and auto-logged in", username=new_username_reg)
                        st.session_state.authenticated = True
                        st.session_state.username = new_username_reg
                        st.balloons()
                        st.experimental_rerun()
                    else:
                        st.error("Username already exists or a database error occurred. Please choose a different username.")

def display_past_analyses():
    st.header("📊 Your Analysis History")
    st.markdown("Review feedback and results from your previous resume optimization sessions.")
    st.divider()
    user_analyses = get_user_analyses(st.session_state.username) # Already returns list of dicts
    if user_analyses:
        for analysis_item in user_analyses:
            expander_title = f"📜 {analysis_item.get('resume_name', 'N/A')} vs {analysis_item.get('job_name', 'N/A')} ({analysis_item.get('analysis_date', 'N/A')[:10]})"
            with st.expander(expander_title):
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.info(f"**Target Role:** {analysis_item.get('target_role', 'N/A')}")
                    st.caption(f"**Analysis ID:** {analysis_item.get('id', 'N/A')}")
                with col_meta2:
                    st.info(f"**Model Used:** {analysis_item.get('model_used', 'N/A')}")
                    st.caption(f"**Language:** {analysis_item.get('language', 'N/A')}")
                st.markdown("**AI Feedback Summary:**")
                feedback_text = analysis_item.get('feedback', '')
                summary = (feedback_text[:350] + "...") if len(feedback_text) > 350 else feedback_text
                st.markdown(f"<blockquote style='border-left: 3px solid #ddd; padding-left: 10px; margin-left: 0; color: #555;'><i>{summary}</i></blockquote>", unsafe_allow_html=True)
    else:
        st.info("No past analyses found. Navigate to 'New Analysis' to get started! 🚀")

def display_analysis_results(current_analysis_results):
    if not current_analysis_results:
        st.warning("No analysis results available to display.", icon="⚠️")
        return

    tab_titles_list = [
        "📊 Summary & Scores", "🔑 Keyword Analysis", "📅 Formatting Check",
        "🔗 Semantic Relevance", "💡 AI Feedback"
    ]
    tab_summary, tab_keywords, tab_formatting, tab_semantic, tab_ai = st.tabs(tab_titles_list)

    with tab_summary:
        st.subheader("Overall Resume Snapshot")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        sim_score_val = current_analysis_results.get('token_sim_score', 0.0)
        missing_keywords_list = current_analysis_results.get('missing_keywords', [])
        date_issues_list_basic = current_analysis_results.get('basic_date_issues', [])
        date_issues_list_adv = current_analysis_results.get('advanced_date_issues', [])

        metric_col1.metric("Similarity Score", f"{sim_score_val:.1%}", help="Keyword overlap with job description.")
        metric_col2.metric("Missing Keywords", len(missing_keywords_list), delta=f"-{len(missing_keywords_list)}" if missing_keywords_list else None, delta_color="inverse" if missing_keywords_list else "normal")
        metric_col3.metric("Date Issues", len(date_issues_list_basic) + len(date_issues_list_adv), delta=f"-{len(date_issues_list_basic) + len(date_issues_list_adv)}" if (date_issues_list_basic or date_issues_list_adv) else None, delta_color="inverse" if (date_issues_list_basic or date_issues_list_adv) else "normal")
        st.divider()
        st.subheader("🗣️ Action Verb Quality")
        action_verb_suggestions_list = suggest_action_verbs(current_analysis_results.get('resume_text', ''))
        if action_verb_suggestions_list:
            st.warning("Consider these action verb enhancements:")
            for suggestion_text in action_verb_suggestions_list:
                st.markdown(f"👉 _{suggestion_text}_")
        else:
            st.success("✅ Your action verbs appear strong and impactful!")

    with tab_keywords:
        st.subheader("🔑 Keyword Gap Analysis")
        MAX_KW_COLS = 12; MAX_KW_HIGHLIGHT = 7
        if missing_keywords_list:
            st.error(f"**{len(missing_keywords_list)} keywords/phrases from the job description seem missing or underutilized.**")
            st.markdown(f"Top {min(len(missing_keywords_list), MAX_KW_COLS)} are shown. Incorporate them naturally:")
            # Display keywords in columns
            num_kw_cols = 3
            for i in range(0, min(len(missing_keywords_list), MAX_KW_COLS), num_kw_cols):
                cols = st.columns(num_kw_cols)
                for j, keyword_item in enumerate(missing_keywords_list[i : i + num_kw_cols]):
                    with cols[j]:
                        st.info(f"`{keyword_item}`")
            if len(missing_keywords_list) > MAX_KW_COLS:
                with st.expander("View all missing keywords..."):
                    st.caption(", ".join(missing_keywords_list))
            st.divider()
            st.subheader("📝 Resume with Missing Keywords Highlighted")
            st.caption(f"(Highlights up to {MAX_KW_HIGHLIGHT} keywords. Focus on natural integration.)")
            highlighted_resume_html = highlight_missing_keywords(current_analysis_results.get('resume_text', ''), missing_keywords_list[:MAX_KW_HIGHLIGHT])
            st.markdown(f'<div style="border:1px solid #e0e0e0; padding:15px; border-radius:8px; background-color:#fdfdfd; max-height:400px; overflow-y:auto;">{highlighted_resume_html}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ Excellent! Your resume covers key terms from the job description well.")

    with tab_formatting:
        st.subheader("📅 Date & Formatting Consistency")
        if date_issues_list_basic or date_issues_list_adv:
            if date_issues_list_basic:
                st.error("🚨 Basic Date Formatting Issues:")
                for issue_line in date_issues_list_basic: st.markdown(f"  ❌ `{issue_line}` (Tip: Use YYYY for years).")
            if date_issues_list_adv:
                st.warning("💡 Advanced Date Formatting Suggestions:")
                for issue_text in date_issues_list_adv: st.markdown(f"  🤔 {issue_text}")
        else:
            st.success("✅ Your date formatting appears consistent and clear!")

    with tab_semantic:
        st.subheader("🔗 Semantic Relevance Analysis")
        st.markdown("How well do your resume statements align with job requirements, beyond exact keywords?")
        semantic_matches_dict = current_analysis_results.get('semantic_results', {})
        MAX_SEMANTIC_TO_SHOW = 5; displayed_sem_count = 0
        if semantic_matches_dict:
            for res_line_text, matches_list_items in list(semantic_matches_dict.items()):
                strong_matches = [m_item for m_item in matches_list_items if m_item[1] > 0.60]
                if not strong_matches or displayed_sem_count >= MAX_SEMANTIC_TO_SHOW: continue
                displayed_sem_count += 1
                exp_title = res_line_text[:60] + "..." if len(res_line_text) > 60 else res_line_text
                with st.expander(f"📌 **Resume:** _{exp_title}_"):
                    st.markdown(f"**Your Statement:**\n> {res_line_text}")
                    st.markdown("**Potential Alignments in Job Description:**")
                    for job_req_text, score_val in strong_matches:
                        clr = "#28a745" if score_val > 0.75 else ("#ffc107" if score_val > 0.65 else "#6c757d")
                        st.markdown(f'<div style="margin-bottom:8px; padding:10px; border-left:4px solid {clr}; background-color:{clr}1A; border-radius:4px;">'
                                    f'<strong style="color:{clr};">Match: {score_val*100:.0f}%</strong><br>{job_req_text}</div>', unsafe_allow_html=True)
            if displayed_sem_count == 0 and semantic_matches_dict:
                st.info("Some semantic links found, but none met the high display threshold (>60%).")
            elif displayed_sem_count < len(semantic_matches_dict) and displayed_sem_count > 0 :
                st.caption(f"Showing top {displayed_sem_count} strongest semantic links.")
        else:
            st.info("No significant semantic matches found or analysis skipped.")

    with tab_ai:
        st.subheader("💡 AI Feedback & Tailoring Suggestions")
        llm_feedback_from_results = current_analysis_results.get('llm_feedback', "No AI feedback available for this analysis.")
        st.markdown(llm_feedback_from_results)
        st.divider()
        st.subheader("✏️ Your Edits & Notes (Optional)")
        current_edited_feedback = current_analysis_results.get('edited_feedback', llm_feedback_from_results)
        feedback_key_unique = f"feedback_edit_{current_analysis_results.get('resume_name', 'r')}_{current_analysis_results.get('job_name', 'j')}_{st.session_state.username}"
        new_edited_feedback = st.text_area(
            "Modify AI feedback or add notes. This version is used for saving and generating the enhanced resume.",
            value=current_edited_feedback, height=300, key=feedback_key_unique,
            help="Changes are noted when you interact with save/generate buttons."
        )
        if new_edited_feedback != current_edited_feedback:
            st.session_state.analysis_results['edited_feedback'] = new_edited_feedback
            # st.toast("Feedback edits updated.", icon="📝") # Subtle notification


# --- Main Application Flow ---
def main():
    init_session_state()
    init_db()

    if not st.session_state.authenticated:
        display_login_register()
        st.stop()

    st.set_page_config(page_title="Resume Optimizer Dashboard", layout="wide", initial_sidebar_state="expanded")

    with st.sidebar:
        if Path(LOGO_PATH).is_file(): st.image(LOGO_PATH, width=100)
        else: st.markdown("### 🧾 Resume Optimizer")
        st.markdown(f"#### Welcome, {st.session_state.username}!")
        st.divider()
        page_selection = st.radio(
            "Navigation", ["🚀 New Analysis", "📊 Past Analyses"], key="main_nav_radio",
            captions=["Analyze a new resume.", "View your saved history."], horizontal=True # Horizontal radio
        )
        st.divider()
        if st.button("Log Out", key="main_logout_btn", type="secondary", use_container_width=True):
            log.info("User logging out", username=st.session_state.username)
            if hasattr(llm_engine, 'unload_current_model'): llm_engine.unload_current_model()
            # Clear all session state keys more robustly
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.experimental_rerun()

        st.divider()
        clear_history_button_key = "clear_history_initial_button"
        if st.button("⚠️ Clear My Analysis History", key=clear_history_button_key, type="default", use_container_width=True, help="Permanently deletes all your saved analyses."):
            # Only set to True if it's not already True to avoid unnecessary reruns if already confirming
            if not st.session_state.get('confirm_delete_history', False):
                st.session_state.confirm_delete_history = True
                st.experimental_rerun() # Rerun to show confirmation

        if st.session_state.get('confirm_delete_history', False):
            st.error("Permanently delete ALL past analyses? This action cannot be undone.")
            col_confirm_del, col_cancel_del = st.columns(2)

            # Unique keys for confirmation buttons
            confirm_delete_button_key = "confirm_delete_yes_button"
            cancel_delete_button_key = "confirm_delete_cancel_button"

            if col_confirm_del.button("Yes, Delete All", key=confirm_delete_button_key, type="primary", use_container_width=True):
                if delete_user_analyses(st.session_state.username):
                    st.success("Your analysis history has been cleared.")
                    st.session_state.analysis_results = None # Clear current view if any
                else:
                    st.error("Could not clear analysis history.") # delete_user_analyses already shows st.error
                st.session_state.confirm_delete_history = False # Reset confirmation state
                st.experimental_rerun() # Rerun to reflect changes and hide confirmation

            if col_cancel_del.button("Cancel Deletion", key=cancel_delete_button_key, use_container_width=True):
                st.session_state.confirm_delete_history = False # Reset confirmation state
                st.experimental_rerun() # Rerun to hide confirmation

        st.caption(f"© {datetime.now().year} Resume Optimizer Agent")

    if page_selection == "📊 Past Analyses":
        display_past_analyses()
    elif page_selection == "🚀 New Analysis":
        st.header("🚀 New Resume Analysis")
        st.markdown("Upload your resume and a job description for AI-powered optimization suggestions. Results appear below after analysis.")
        st.divider()

        with st.form(key="main_analysis_input_form"):
            st.subheader("📄 Upload Documents")
            form_col_resume, form_col_job = st.columns(2)
            with form_col_resume: resume_file_upload = st.file_uploader("1. Your Resume (PDF)", type="pdf", key="form_resume_uploader")
            with form_col_job: job_file_upload = st.file_uploader("2. Job Description (PDF or TXT)", type=["pdf", "txt"], key="form_job_uploader")
            st.divider()
            st.subheader("⚙️ Analysis Configuration")
            form_col_config1, form_col_config2 = st.columns(2)
            with form_col_config1:
                target_role_val = st.text_input("3. Target Job Title/Industry (Optional):", placeholder="e.g., AI Engineer", key="form_target_role", help="Tailors AI feedback.")
                language_val = st.selectbox("4. Document Language:", ["English", "French", "Spanish", "German", "Other"], key="form_language", help="Primary language of documents.")
            with form_col_config2:
                st.markdown("**🤖 LLM Model Selection**")
                rec_target = target_role_val
                rec_size = "medium"
                if resume_file_upload and hasattr(resume_file_upload, 'size'): rec_size = "short" if resume_file_upload.size < 60000 else ("long" if resume_file_upload.size > 180000 else "medium")
                model_recs = get_model_recommendations(rec_target, rec_size)
                st.info(f"Recommended: `{model_recs.get('primary', DEFAULT_LLM_MODEL)}`\n_{model_recs.get('reason', '')}_", icon="💡")
                available_models = llm_engine.get_available_model_names()
                try: default_idx = available_models.index(model_recs.get('primary'))
                except (ValueError, AttributeError, IndexError): default_idx = 0
                model_choice_val = st.selectbox("5. Choose LLM Model:", available_models, index=default_idx, key="form_model_choice", help="Recommended model is pre-selected.")
            st.divider()
            submit_analysis_button = st.form_submit_button("✨ Analyze Resume & Get Feedback", type="primary", use_container_width=True)

        if submit_analysis_button:
            if resume_file_upload and job_file_upload:
                log.info("Analysis form processing", user=st.session_state.username, resume=resume_file_upload.name, job=job_file_upload.name)
                with st.spinner(f"🚀 Launching analysis with '{model_choice_val}'... This may take a few moments."):
                    try:
                        llm_engine.get_llm_instance(model_choice_val)
                        log.info(f"AI Model {model_choice_val} is ready.")
                    except Exception as model_err:
                        st.error(f"Failed to prepare AI model '{model_choice_val}': {model_err}")
                        log.critical("AI Model loading failed", model=model_choice_val, error=str(model_err), exc_info=True)
                        st.stop()

                # Pass tuples of bytes to @st.cache_data function
                # This is a workaround for caching mutable byte-like objects if Streamlit has issues
                # Directly passing getvalue() results should be fine but this is safer for cache integrity.
                _resume_bytes_for_cache = (resume_file_upload.getvalue(),)
                _job_bytes_for_cache = (job_file_upload.getvalue(),)

                analysis_run_output = run_full_analysis(
                    _resume_bytes_for_cache, _job_bytes_for_cache,
                    resume_file_upload.name, job_file_upload.name,
                    model_choice_val, target_role_val, language_val
                )
                st.session_state.analysis_results = analysis_run_output
                if analysis_run_output: st.success("Analysis complete! Results are below.", icon="🎉")
                else: st.error("Analysis could not be completed. Please check input files or try a different model.", icon="🆘")
            else:
                st.warning("⚠️ Please upload both your resume and the job description.", icon="❗")

    if st.session_state.analysis_results and page_selection == "🚀 New Analysis":
        st.divider()
        st.header("📈 Analysis Results Breakdown")
        display_analysis_results(st.session_state.analysis_results)
        st.divider()
        st.header("🚀 Next Steps & Output Generation")
        results_for_actions = st.session_state.analysis_results
        action_col_save, action_col_generate = st.columns(2)
        with action_col_save:
            st.subheader("💾 Save Analysis")
            if st.button("Save to My History", key="action_save_btn", help="Saves current analysis and feedback."):
                feedback_final_to_save = results_for_actions.get('edited_feedback', results_for_actions.get('llm_feedback', ''))
                data_for_db = results_for_actions.copy()
                data_for_db['llm_feedback_to_save'] = feedback_final_to_save
                if save_analysis(st.session_state.username, data_for_db): st.success("Analysis saved to history!")
                else: st.error("Failed to save analysis.")
        with action_col_generate:
            st.subheader("📄 Generate Enhanced PDF")
            st.markdown("Use AI feedback to create an improved resume version.")
            feedback_for_pdf_gen = results_for_actions.get('edited_feedback', results_for_actions.get('llm_feedback', ''))
            pdf_gen_disabled = not feedback_for_pdf_gen.strip() or \
                               feedback_for_pdf_gen.startswith("Error:") or \
                               feedback_for_pdf_gen.startswith("⚠️")
            if pdf_gen_disabled: st.caption("PDF generation disabled: AI feedback missing or has errors.")
            if st.button("✨ Rewrite & Generate PDF", key="action_generate_pdf_btn", type="primary", disabled=pdf_gen_disabled, help="AI rewrites resume based on feedback, then generates PDF."):
                with st.spinner("⚙️ Applying feedback & crafting new resume PDF... This is intensive!"):
                    try:
                        final_updated_resume_text = generate_enhanced_resume(
                            original_resume=results_for_actions.get('resume_text', ''),
                            feedback=feedback_for_pdf_gen,
                            model_name=results_for_actions.get('model_choice', DEFAULT_LLM_MODEL),
                            target_role=results_for_actions.get('target_role', ''),
                            language=results_for_actions.get('language', 'English')
                        )
                        if final_updated_resume_text.startswith("⚠️ Error"):
                            st.error(f"LLM failed to rewrite resume: {final_updated_resume_text}")
                            log.error("LLM resume rewrite failed by generate_enhanced_resume", detail=final_updated_resume_text)
                        else:
                            st.info("Generating professionally formatted PDF...")
                            final_pdf_bytes = generate_professional_pdf_bytes(
                                final_updated_resume_text,
                                title=f"Enhanced-{results_for_actions.get('resume_name', 'resume').replace('.pdf','')}"
                            )
                            if final_pdf_bytes:
                                st.download_button(
                                    label="⬇️ Download Enhanced Resume PDF", data=final_pdf_bytes,
                                    file_name=f"Enhanced_{results_for_actions.get('resume_name', 'resume').replace('.pdf','')}.pdf",
                                    mime="application/pdf"
                                )
                                log.info("Enhanced PDF ready for download", user=st.session_state.username)
                            else: st.error("Failed to generate PDF from rewritten text. Fallback PDF might have been attempted.")
                    except Exception as gen_err:
                        log.exception("PDF generation/rewrite process failed critically", exc_info=True)
                        st.error(f"Error during PDF generation: {gen_err}")

if __name__ == "__main__":
    # Ensure .env is loaded before anything else that might need env vars
    load_dotenv() # Moved here to be absolutely first if main() is entry point
    main()
