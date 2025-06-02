import hashlib
import streamlit as st
import sqlite3
import os
import io
import re
import logging
import structlog
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from passlib.context import CryptContext
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# --- Local Imports ---
from parser import extract_text_from_pdf
from llm_engine import analyze_resume, generate_enhanced_resume, get_model_recommendations
from utils import (
    compute_token_similarity,
    find_missing_keywords,
    check_date_formatting,
    advanced_date_formatting_check,
    highlight_missing_keywords,
    suggest_action_verbs,
    calculate_keyword_density
)
from semantic_search import SemanticMatcher, create_semantic_matcher

# --- Configuration & Initialization ---
load_dotenv()  # Load variables from .env file

# Configure structured logging
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
        structlog.dev.ConsoleRenderer() # Use ConsoleRenderer for development
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
)

# --- Constants & Environment Variables ---
DATABASE_NAME = os.getenv("DATABASE_NAME", "resume_optimizer.db")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache/semantic_indexes"))
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "mistral:latest")
PROMPT_DIR = Path("prompts")
ENHANCED_PROMPT_FILE = PROMPT_DIR / "enhanced_resume_review.txt"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR.mkdir(parents=True, exist_ok=True)

# Password hashing context using passlib (more secure than plain SHA256)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'authenticated': False,
        'username': None,
        'cached_job_index': None,
        'cached_job_hash': None,
        'analysis_results': None # Store results to avoid re-computation on rerun
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Database Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row # Return rows as dict-like objects
        log.debug("Database connection established", db_name=DATABASE_NAME)
        return conn
    except sqlite3.Error as e:
        log.error("Database connection failed", error=str(e), db_name=DATABASE_NAME)
        st.error(f"Database error: {e}")
        return None

def init_db():
    """Initializes the database tables if they don't exist."""
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
                                FOREIGN KEY (username) REFERENCES users (username)
                             )""")
            log.info("Database initialized successfully")
        except sqlite3.Error as e:
            log.error("Database initialization failed", error=str(e))
        finally:
            conn.close()

# --- Authentication Functions (Using Passlib) ---
def verify_password(plain_password, hashed_password):
    """Verifies a plain password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    """Hashes a password using the configured context."""
    return pwd_context.hash(password)

def authenticate_user(username, password):
    """Authenticates a user against the database."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and verify_password(password, user['password_hash']):
            log.info("User authenticated successfully", username=username)
            return True
        log.warning("Authentication failed", username=username)
        return False
    except sqlite3.Error as e:
        log.error("Authentication database error", error=str(e), username=username)
        return False
    finally:
        conn.close()

def register_user(username, password):
    """Registers a new user in the database."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn:
            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                         (username, hash_password(password)))
        log.info("User registered successfully", username=username)
        return True
    except sqlite3.IntegrityError:
        log.warning("Registration failed: Username already exists", username=username)
        return False # Username already exists
    except sqlite3.Error as e:
        log.error("Registration database error", error=str(e), username=username)
        return False
    finally:
        conn.close()

# --- Analysis Data Functions ---
def save_analysis(username, analysis_data):
    """Saves analysis results to the database."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn:
            conn.execute("""INSERT INTO analyses (username, resume_name, job_name, analysis_date,
                                             feedback, similarity_score, missing_keywords_count,
                                             date_issues_count, model_used, target_role, language)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (username,
                          analysis_data.get('resume_name', 'N/A'),
                          analysis_data.get('job_name', 'N/A'),
                          datetime.now().isoformat(),
                          analysis_data.get('llm_feedback', ''),
                          analysis_data.get('token_sim_score', 0.0),
                          len(analysis_data.get('missing_keywords', [])),
                          len(analysis_data.get('basic_date_issues', [])) + len(analysis_data.get('advanced_date_issues', [])),
                          analysis_data.get('model_choice', 'N/A'),
                          analysis_data.get('target_role', ''),
                          analysis_data.get('language', 'N/A')))
        log.info("Analysis saved successfully", username=username, resume=analysis_data.get('resume_name'))
        return True
    except sqlite3.Error as e:
        log.error("Failed to save analysis", error=str(e), username=username)
        st.error(f"Failed to save analysis: {e}")
        return False
    finally:
        conn.close()

def get_user_analyses(username):
    """Retrieves past analyses for a given user."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.execute("""SELECT id, resume_name, job_name, analysis_date, feedback, model_used, target_role
                                 FROM analyses WHERE username = ? ORDER BY analysis_date DESC""", (username,))
        results = cursor.fetchall()
        log.debug("Retrieved past analyses", username=username, count=len(results))
        return results
    except sqlite3.Error as e:
        log.error("Failed to retrieve analyses", error=str(e), username=username)
        return []
    finally:
        conn.close()

# --- PDF Generation ---
def text_to_pdf_bytes(resume_text: str) -> bytes:
    """Converts plain text resume to PDF bytes using ReportLab."""
    buffer = io.BytesIO()
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        x_margin, y_margin = 40, 40
        line_height = 14
        c.setFont("Helvetica", 11) # Slightly smaller font
        y = height - y_margin

        for line in resume_text.splitlines():
            if y < y_margin:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - y_margin
            c.drawString(x_margin, y, line)
            y -= line_height

        c.save()
        pdf_bytes = buffer.getvalue()
        log.info("PDF generated successfully from text")
        return pdf_bytes
    except Exception as e:
        log.error("PDF generation failed", error=str(e))
        st.error(f"Error generating PDF: {e}")
        return b""
    finally:
        buffer.close()

# --- Core Analysis Logic ---
@st.cache_data(show_spinner=False) # Cache the main analysis function
def run_full_analysis(_resume_bytes, _job_bytes, resume_name, job_name, model_choice, target_role, language):
    """Runs the full analysis pipeline. Underscores prevent hashing large byte objects."""
    analysis_results = {}
    analysis_results['resume_name'] = resume_name
    analysis_results['job_name'] = job_name
    analysis_results['model_choice'] = model_choice
    analysis_results['target_role'] = target_role
    analysis_results['language'] = language

    progress_bar = st.progress(0, text="Starting analysis...")
    steps = 7

    try:
        # Step 1: Extract resume text
        progress_bar.progress(1/steps, text="📄 Extracting text from resume...")
        resume_text = extract_text_from_pdf(io.BytesIO(_resume_bytes))
        analysis_results['resume_text'] = resume_text
        log.debug("Resume text extracted", length=len(resume_text))

        # Step 2: Extract job text
        progress_bar.progress(2/steps, text="📋 Processing job description...")
        job_file_obj = io.BytesIO(_job_bytes)
        if job_name.lower().endswith('.txt'):
            job_text = job_file_obj.read().decode("utf-8")
        else:
            job_text = extract_text_from_pdf(job_file_obj)
        analysis_results['job_text'] = job_text
        log.debug("Job description text extracted", length=len(job_text))

        # Step 3: Semantic Matcher Initialization & Indexing (with caching)
        progress_bar.progress(3/steps, text="🧠 Building/Loading semantic index...")
        job_hash = hashlib.md5(job_text.encode()).hexdigest()
        if st.session_state.cached_job_hash != job_hash or st.session_state.cached_job_index is None:
            log.info("Cache miss or invalid. Building new semantic index.", job_hash=job_hash)
            job_sentences = [line.strip() for line in job_text.splitlines() if len(line.strip()) > 10]
            if not job_sentences:
                st.warning("Could not extract sentences from job description for semantic analysis.")
                sem_matcher = None
            else:
                sem_matcher = create_semantic_matcher(language=language)
                sem_matcher.build_index(job_sentences)
                st.session_state.cached_job_index = sem_matcher
                st.session_state.cached_job_hash = job_hash
        else:
            log.info("Using cached semantic index.", job_hash=job_hash)
            sem_matcher = st.session_state.cached_job_index
        analysis_results['semantic_matcher'] = sem_matcher

        # Step 4: Perform Basic Analyses
        progress_bar.progress(4/steps, text="🔍 Analyzing similarities & keywords...")
        analysis_results['token_sim_score'] = compute_token_similarity(resume_text, job_text)
        analysis_results['missing_keywords'] = find_missing_keywords(resume_text, job_text)
        analysis_results['basic_date_issues'] = check_date_formatting(resume_text)
        analysis_results['advanced_date_issues'] = advanced_date_formatting_check(resume_text)
        log.debug("Basic analyses complete", score=analysis_results['token_sim_score'], missing=len(analysis_results['missing_keywords']))

        # Step 5: Semantic Matching
        progress_bar.progress(5/steps, text="🎯 Finding semantic matches...")
        if sem_matcher:
            resume_sentences = [line.strip() for line in resume_text.splitlines() if len(line.strip()) > 10]
            if resume_sentences:
                analysis_results['semantic_results'] = sem_matcher.query(resume_sentences, top_k=2)
                log.debug("Semantic matching complete")
            else:
                analysis_results['semantic_results'] = {}
                st.warning("Could not extract sentences from resume for semantic analysis.")
        else:
             analysis_results['semantic_results'] = {}

        # Step 6: Generate LLM Feedback using Enhanced Prompt
        progress_bar.progress(6/steps, text="🤖 Generating AI feedback...")
        if not ENHANCED_PROMPT_FILE.exists():
            st.error(f"Prompt file not found: {ENHANCED_PROMPT_FILE}")
            log.error("Prompt file missing", path=str(ENHANCED_PROMPT_FILE))
            analysis_results['llm_feedback'] = "Error: Prompt file missing."
        else:
            with open(ENHANCED_PROMPT_FILE, "r") as f:
                prompt_template = f.read()

            # Format the prompt with context
            formatted_prompt = prompt_template.format(
                target_role=target_role if target_role else "Not Specified",
                language=language,
                resume=resume_text,
                job=job_text
            )

            analysis_results['llm_feedback'] = analyze_resume(
                resume_text, # Already included in formatted_prompt
                job_text,    # Already included in formatted_prompt
                formatted_prompt, # Pass the fully formatted prompt
                model_name=model_choice,
                target_role=target_role, # Pass context separately too if needed by llm_engine
                language=language
            )
            log.debug("LLM feedback generated", model=model_choice)

        # Step 7: Complete
        progress_bar.progress(7/steps, text="✅ Analysis complete!")
        log.info("Full analysis completed successfully", resume=resume_name, job=job_name)
        st.session_state.analysis_results = analysis_results # Store results
        return analysis_results

    except Exception as e:
        log.exception("Analysis failed", exc_info=True)
        st.error(f"An error occurred during analysis: {e}")
        progress_bar.empty()
        return None
    finally:
        # Ensure progress bar is removed even if analysis is cached
        progress_bar.empty()

# --- Streamlit UI Functions ---
def display_login_register():
    """Displays the login and registration forms."""
    st.title("🧾 Resume Optimizer Agent")
    st.subheader("🔐 Authentication Required")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")
            if login_submitted:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    log.info("User logged in", username=username)
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password!")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submitted = st.form_submit_button("Register")
            if register_submitted:
                if not new_username or not new_password:
                     st.error("Username and password cannot be empty.")
                elif new_password != confirm_password:
                    st.error("Passwords don't match!")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters long.")
                elif register_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists or database error occurred.")

def display_past_analyses():
    """Displays the user's past analysis results."""
    st.header("📊 Your Past Analyses")
    analyses = get_user_analyses(st.session_state.username)
    if analyses:
        for i, analysis in enumerate(analyses):
            expander_title = f"Analysis {analysis['id']}: {analysis['resume_name']} vs {analysis['job_name']} ({analysis['analysis_date'][:10]})"
            with st.expander(expander_title):
                st.write(f"**Target Role:** {analysis['target_role'] if analysis['target_role'] else 'N/A'}")
                st.write(f"**Model Used:** {analysis['model_used']}")
                st.write(f"**Date:** {analysis['analysis_date']}")
                st.markdown("**Feedback:**")
                st.markdown(analysis['feedback']) # Display as Markdown
    else:
        st.info("No past analyses found. Create your first analysis!")

def display_analysis_results(results):
    """Displays the results of a new analysis in tabs."""
    if not results:
        return

    tab_titles = [
        "📊 Overview",
        "🔤 Keywords",
        "📅 Formatting",
        "🎯 Semantic Matches",
        "🤖 AI Feedback"
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

    with tab1:
        st.subheader("📊 Analysis Overview")
        col1, col2, col3 = st.columns(3)
        sim_score = results.get('token_sim_score', 0.0)
        missing_kw_count = len(results.get('missing_keywords', []))
        date_issues_count = len(results.get('basic_date_issues', [])) + len(results.get('advanced_date_issues', []))

        col1.metric("Similarity Score", f"{sim_score:.2f}", delta=None, help="Cosine similarity based on word overlap.")
        col2.metric("Missing Keywords", missing_kw_count, help="Keywords from job description not found in resume.")
        col3.metric("Date Issues", date_issues_count, help="Potential inconsistencies in date formatting.")

        # Add Action Verb Suggestions
        st.subheader("🗣️ Action Verb Suggestions")
        action_verb_suggestions = suggest_action_verbs(results.get('resume_text', ''))
        if action_verb_suggestions:
            for suggestion in action_verb_suggestions:
                st.info(f"💡 {suggestion}")
        else:
            st.success("✅ Action verbs look strong!")

        # Add Keyword Density (Optional - can be slow)
        # st.subheader("🔑 Keyword Density")
        # top_keywords = results.get('missing_keywords', [])[:5] # Density for top 5 missing
        # if top_keywords:
        #     density = calculate_keyword_density(results.get('resume_text', ''), top_keywords)
        #     st.table(density)

    with tab2:
        st.subheader("🔤 Keyword Analysis")
        missing_keywords = results.get('missing_keywords', [])
        if missing_keywords:
            st.warning(f"Found {len(missing_keywords)} missing keywords/phrases:")
            # Display keywords more clearly
            kw_cols = st.columns(4)
            for i, keyword in enumerate(missing_keywords[:20]): # Show more keywords
                kw_cols[i % 4].code(keyword)

            if len(missing_keywords) > 20:
                with st.expander(f"Show all {len(missing_keywords)} missing keywords"):
                    st.text(", ".join(missing_keywords))

            st.subheader("📝 Resume with Highlighted Missing Keywords")
            st.caption("(Highlights first 10 missing keywords for clarity)")
            highlighted_resume = highlight_missing_keywords(results.get('resume_text', ''), missing_keywords)
            st.markdown(highlighted_resume, unsafe_allow_html=True)
        else:
            st.success("✅ Your resume appears to cover the key terms from the job description!")

    with tab3:
        st.subheader("📅 Date Formatting Analysis")
        basic_issues = results.get('basic_date_issues', [])
        advanced_issues = results.get('advanced_date_issues', [])
        if basic_issues or advanced_issues:
            if basic_issues:
                st.warning("Potential basic date formatting issues (Month without 4-digit year?):")
                for line in basic_issues:
                    st.text(f"❌ {line}")
            if advanced_issues:
                st.info("Advanced date formatting suggestions:")
                for issue in advanced_issues:
                    st.markdown(f"💡 {issue}") # Use markdown for potential formatting
        else:
            st.success("✅ No significant date formatting issues detected!")

    with tab4:
        st.subheader("🎯 Semantic Matches")
        st.caption("Shows resume sentences with their closest semantic matches in the job description.")
        semantic_results = results.get('semantic_results', {})
        if semantic_results:
            match_count = 0
            for res_line, matches in list(semantic_results.items())[:10]: # Show top 10 matches
                match_count += 1
                with st.expander(f"Match {match_count}: " + (res_line[:60] + "..." if len(res_line) > 60 else res_line)):
                    st.markdown(f"**Resume Line:**> {res_line}")
                    st.markdown("**Potential Job Requirement Matches:**")
                    for job_req, score in matches:
                        color = "#28a745" if score > 0.75 else "#ffc107" if score > 0.6 else "#dc3545"
                        st.markdown(f'<span style="color:{color}">●</span> **(Score: {score:.2f})** {job_req}', unsafe_allow_html=True)
        else:
            st.info("No significant semantic matches found or semantic analysis was skipped.")

    with tab5:
        st.subheader("🤖 AI-Generated Feedback")
        llm_feedback = results.get('llm_feedback', "No feedback generated.")
        st.markdown(llm_feedback) # Display feedback as Markdown

        # Editable feedback area
        st.subheader("✏️ Refine Feedback (Optional)")
        edited_feedback = st.text_area(
            "You can modify the AI's feedback here before generating the updated resume:",
            value=llm_feedback,
            height=250,
            key=f"edit_feedback_{results.get('resume_name')}_{results.get('job_name')}", # Unique key
            help="Edit the AI feedback to customize your resume update. This edited version will be saved and used for generation."
        )
        results['edited_feedback'] = edited_feedback # Store edited feedback

# --- Main Application Flow ---
def main():
    """Main function to run the Streamlit application."""
    init_session_state()
    init_db() # Ensure DB is initialized

    if not st.session_state.authenticated:
        display_login_register()
        st.stop()

    # --- Authenticated App UI ---
    st.sidebar.success(f"Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        log.info("User logged out", username=st.session_state.username)
        # Clear sensitive session state on logout
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    page = st.sidebar.selectbox("Navigate", ["🚀 New Analysis", "📊 Past Analyses"], key="navigation")

    if page == "📊 Past Analyses":
        display_past_analyses()
        st.stop()

    # --- New Analysis Page ---
    st.header("🚀 New Resume Analysis")

    # Input Form
    with st.form(key="analysis_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            resume_file = st.file_uploader("1. Upload your resume (PDF)", type="pdf", key="resume_upload")
            job_file = st.file_uploader("2. Upload job description (PDF or TXT)", type=["pdf", "txt"], key="job_upload")

        with col2:
            # Model recommendations
            st.write("**Model Recommendation:**")
            rec_target = st.session_state.get('target_role_input', '')
            recommendations = get_model_recommendations(rec_target)
            st.info(f"**Recommended:** `{recommendations.get('primary', DEFAULT_LLM_MODEL)}`\n*{recommendations.get('reason', '')}*", icon="💡")

            model_options = list(get_model_recommendations().keys()) # Assuming llm_engine provides available models
            model_choice = st.selectbox(
                "3. Choose LLM Model:",
                options=[
                    "mistral:latest", "llama3.1:latest", "deepseek-r1:1.5b",
                    "gemma:7b", "codellama:7b"
                ],
                index=0,
                key="model_select",
                help="Select the AI model. Recommendations provided above."
            )
            target_role = st.text_input(
                "4. Target Job Title/Industry (Optional):",
                placeholder="e.g., Senior Data Scientist",
                key="target_role_input",
                help="Specifying helps tailor the AI feedback."
            )
            language = st.selectbox(
                "5. Resume Language:",
                options=["English", "French", "Spanish", "German", "Other"],
                key="language_select",
                help="Select the primary language of your documents."
            )

        submitted = st.form_submit_button("✨ Analyze Resume", type="primary")

    # --- Analysis Execution and Display ---
    if submitted and resume_file and job_file:
        log.info("Analysis submitted", user=st.session_state.username, resume=resume_file.name, job=job_file.name)
        # Read file bytes once
        resume_bytes = resume_file.getvalue()
        job_bytes = job_file.getvalue()
        # Run analysis (will use cache if inputs are the same)
        analysis_results = run_full_analysis(
            resume_bytes,
            job_bytes,
            resume_file.name,
            job_file.name,
            model_choice,
            target_role,
            language
        )
        st.session_state.analysis_results = analysis_results # Update session state

    # Display results if they exist in session state
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.divider()
        st.header("📈 Analysis Results")
        display_analysis_results(results)

        # --- Post-Analysis Actions ---
        st.divider()
        st.header("💾 Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Analysis Results"):
                if save_analysis(st.session_state.username, results):
                    st.success("Analysis saved successfully!")
                else:
                    st.error("Failed to save analysis.")

        with col2:
            st.write("**📄 Generate Updated Resume PDF**")
            generate_pdf_button = st.button("🚀 Generate PDF", type="secondary")

            if generate_pdf_button:
                feedback_to_use = results.get('edited_feedback', results.get('llm_feedback', ''))
                if not feedback_to_use:
                    st.warning("No feedback available to generate updated resume.")
                else:
                    with st.spinner("🔄 Rewriting resume and generating PDF..."):
                        try:
                            updated_resume_text = generate_enhanced_resume(
                                original_resume=results.get('resume_text', ''),
                                feedback=feedback_to_use,
                                model_name=results.get('model_choice', DEFAULT_LLM_MODEL),
                                target_role=results.get('target_role', ''),
                                language=results.get('language', 'English')
                            )

                            if updated_resume_text.startswith("⚠️ Error"):
                                st.error(f"Failed to rewrite resume: {updated_resume_text}")
                                log.error("Resume rewrite failed", error=updated_resume_text)
                            else:
                                pdf_bytes = text_to_pdf_bytes(updated_resume_text)
                                if pdf_bytes:
                                    st.download_button(
                                        label="⬇️ Download Updated Resume PDF",
                                        data=pdf_bytes,
                                        file_name=f"updated_{results.get('resume_name', 'resume')}.pdf",
                                        mime="application/pdf",
                                    )
                                    log.info("Updated PDF ready for download", user=st.session_state.username)
                        except Exception as e:
                            log.exception("PDF generation/rewrite failed", exc_info=True)
                            st.error(f"An error occurred during PDF generation: {e}")

    elif submitted:
        st.warning("Please upload both resume and job description files.")

if __name__ == "__main__":
    main()

