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
# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
# Keep basic canvas for fallback or simple PDFs if generate_professional_pdf_bytes is complex
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

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
    calculate_keyword_density
    # generate_professional_pdf_bytes # Now defined in this file
)
from semantic_search import SemanticMatcher, create_semantic_matcher

# --- Configuration & Initialization ---
load_dotenv()  # Load variables from .env file

# Configure structured logging (structlog)
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
log = structlog.get_logger() # Get a structlog logger

# Basic logging configuration (can be overridden by structlog for its own messages)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(name)s - %(levelname)s - %(message)s", # More informative basic format
)


# --- Constants & Environment Variables ---
DATABASE_NAME = os.getenv("DATABASE_NAME", "resume_optimizer.db")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache/semantic_indexes"))
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "mistral:latest")
PROMPT_DIR = Path("prompts")
ENHANCED_PROMPT_FILE = PROMPT_DIR / "enhanced_resume_review.txt"
LOGO_PATH = "logo.png" # Define path for logo, place logo.png in your project root

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR.mkdir(parents=True, exist_ok=True)

# Password hashing context using passlib
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'authenticated': False,
        'username': None,
        # 'cached_job_index': None, # Replaced by cached_job_semantic_matcher
        # 'cached_job_hash': None,  # Replaced by cached_job_hash_for_matcher
        'analysis_results': None,
        'cached_job_semantic_matcher': None,
        'cached_job_hash_for_matcher': None,
        'confirm_delete_history': False # For delete confirmation checkbox
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Database Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row
        log.debug("Database connection established", db_name=DATABASE_NAME)
        return conn
    except sqlite3.Error as e:
        log.error("Database connection failed", error=str(e), db_name=DATABASE_NAME)
        st.error(f"A database connection error occurred. Please try again later.")
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
            log.info("Database initialized successfully or already exists.")
        except sqlite3.Error as e:
            log.error("Database initialization failed", error=str(e))
        finally:
            conn.close()

def delete_user_analyses(username):
    """Deletes all past analyses for a given user."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn:
            conn.execute("DELETE FROM analyses WHERE username = ?", (username,))
        log.info("All analyses deleted successfully", username=username)
        return True
    except sqlite3.Error as e:
        log.error("Failed to delete analyses", error=str(e), username=username)
        st.error(f"Failed to delete analyses: {e}")
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
        user = cursor.fetchone()
        if user and verify_password(password, user['password_hash']):
            log.info("User authenticated successfully", username=username)
            return True
        log.warning("Authentication failed", username=username, reason="Invalid credentials" if user else "User not found")
        return False
    except sqlite3.Error as e:
        log.error("Authentication database error", error=str(e), username=username)
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
        log.info("User registered successfully", username=username)
        return True
    except sqlite3.IntegrityError: # More specific error for existing username
        log.warning("Registration failed: Username already exists", username=username)
        return False
    except sqlite3.Error as e:
        log.error("Registration database error", error=str(e), username=username)
        return False
    finally:
        conn.close()

# --- Analysis Data Functions ---
def save_analysis(username, analysis_data):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn:
            # Using 'llm_feedback_to_save' as per previous logic
            feedback_to_save = analysis_data.get('llm_feedback_to_save', analysis_data.get('llm_feedback', ''))
            conn.execute("""INSERT INTO analyses (username, resume_name, job_name, analysis_date,
                                             feedback, similarity_score, missing_keywords_count,
                                             date_issues_count, model_used, target_role, language)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (username,
                          analysis_data.get('resume_name', 'N/A'),
                          analysis_data.get('job_name', 'N/A'),
                          datetime.now().isoformat(),
                          feedback_to_save,
                          analysis_data.get('token_sim_score', 0.0),
                          len(analysis_data.get('missing_keywords', [])),
                          len(analysis_data.get('basic_date_issues', [])) + len(analysis_data.get('advanced_date_issues', [])),
                          analysis_data.get('model_choice', 'N/A'),
                          analysis_data.get('target_role', ''),
                          analysis_data.get('language', 'N/A')))
        log.info("Analysis saved successfully", username=username, resume_name=analysis_data.get('resume_name'))
        return True
    except sqlite3.Error as e:
        log.error("Failed to save analysis", error=str(e), username=username)
        st.error(f"Failed to save analysis: {e}")
        return False
    finally:
        conn.close()

def get_user_analyses(username):
    conn = get_db_connection()
    if not conn: return []
    try:
        cursor = conn.execute("""SELECT id, resume_name, job_name, analysis_date, feedback, model_used, target_role, language
                                 FROM analyses WHERE username = ? ORDER BY analysis_date DESC""", (username,))
        results = cursor.fetchall() # Returns list of Row objects
        log.debug("Retrieved past analyses", username=username, count=len(results))
        return results
    except sqlite3.Error as e:
        log.error("Failed to retrieve analyses", error=str(e), username=username)
        return []
    finally:
        conn.close()

# --- PDF Generation ---
def text_to_pdf_bytes_basic(resume_text: str) -> bytes: # Renamed for clarity as fallback
    """Converts plain text resume to a very basic PDF using ReportLab Canvas."""
    buffer = io.BytesIO()
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        textobject = c.beginText(0.75*inch, letter[1] - 0.75*inch) # Start 0.75 inch from top-left
        textobject.setFont("Helvetica", 10) # Smaller font for more content
        textobject.setLeading(12) # Line spacing

        for line in resume_text.splitlines():
            textobject.textLine(line)
            if textobject.getY() < 0.75*inch : # Check if near bottom margin
                c.drawText(textobject)
                c.showPage()
                textobject = c.beginText(0.75*inch, letter[1] - 0.75*inch)
                textobject.setFont("Helvetica", 10)
                textobject.setLeading(12)

        c.drawText(textobject)
        c.save()
        pdf_bytes = buffer.getvalue()
        log.info("Basic PDF generated successfully from text")
        return pdf_bytes
    except Exception as e:
        log.error("Basic PDF generation failed", error=str(e), exc_info=True)
        st.error(f"Error generating basic PDF: {e}")
        return b""
    finally:
        buffer.close()

def generate_professional_pdf_bytes(resume_text: str, title: str = "Resume") -> bytes:
    """
    Generates a more professionally formatted PDF from plain text resume using ReportLab Platypus.
    Tries to infer sections and apply basic styling.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer,
                            pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch,
                            title=title)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='ResumeTitle', parent=styles['h1'], fontName='Helvetica-Bold', fontSize=16, spaceAfter=0.2*inch, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['h2'], fontName='Helvetica-Bold', fontSize=12, spaceBefore=0.2*inch, spaceAfter=0.05*inch, textColor=HexColor('#2E7D32'), keepWithNext=1))
    styles.add(ParagraphStyle(name='BodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=12, spaceAfter=0.05*inch, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='BulletPoint', parent=styles['BodyText'], leftIndent=0.25*inch, bulletIndent=0.1*inch, firstLineIndent=-0.05*inch, spaceAfter=0.02*inch))

    story = []
    section_keywords = ['SUMMARY', 'PROFILE', 'OBJECTIVE', 'EXPERIENCE', 'EMPLOYMENT', 'EDUCATION', 'SKILLS', 'PROJECTS', 'ACTIVITIES', 'AWARDS', 'CERTIFICATIONS', 'REFERENCES', 'TECHNICAL SKILLS', 'PROFESSIONAL EXPERIENCE', 'WORK EXPERIENCE', 'CONTACT', 'LINKS']
    lines = resume_text.splitlines()

    if not lines:
        story.append(Paragraph("No content provided for the resume.", styles['BodyText']))
        doc.build(story)
        return buffer.getvalue()

    first_content_line_index = 0
    # Attempt to make the first non-empty line(s) the title/contact info if they look like it
    potential_title_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if potential_title_lines: # End of title block if empty line after some content
                break
            continue # Skip leading empty lines
        # Heuristic: if line is short, centered-ish (few words), or contains email/phone
        if len(stripped.split()) < 7 or "@" in stripped or bool(re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', stripped)):
            potential_title_lines.append(stripped)
            first_content_line_index = i + 1
        else: # First line that doesn't look like title info
            if not potential_title_lines: # If no title lines found yet, this is the first content
                 potential_title_lines.append(stripped) # Treat first line as title anyway
                 first_content_line_index = i + 1
            break # Stop collecting title lines

    if potential_title_lines:
        for i, title_line in enumerate(potential_title_lines):
            # Main name usually first, make it bigger
            style = styles['ResumeTitle'] if i == 0 and len(potential_title_lines) > 1 else ParagraphStyle(name=f'SubTitle{i}', parent=styles['Normal'], fontSize=10 if "@" in title_line or "linkedin.com" in title_line else 11, alignment=TA_CENTER, spaceAfter=0.05*inch if i < len(potential_title_lines)-1 else 0.15*inch)
            story.append(Paragraph(title_line, style))
    else:
        story.append(Paragraph(title if title else "Resume", styles['ResumeTitle']))


    for line_idx, line in enumerate(lines[first_content_line_index:]):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        is_section_header = False
        # Check if the line matches common section headers (often all caps or starting with keyword)
        upper_stripped_line = stripped_line.upper()
        for keyword in section_keywords:
            if upper_stripped_line.startswith(keyword) and len(stripped_line) < 40:
                is_section_header = True
                break
        if not is_section_header and stripped_line.isupper() and len(stripped_line.split()) < 5 and len(stripped_line) > 2:
             is_section_header = True

        if is_section_header:
            story.append(Paragraph(stripped_line, styles['SectionHeader']))
        elif stripped_line.startswith(('-', '•', '*', '>', 'o', '▪')) and len(stripped_line) > 1:
            bullet_text = stripped_line[1:].strip()
            if bullet_text: # Ensure there's text after the bullet
                story.append(Paragraph(bullet_text, styles['BulletPoint'], bulletText='•'))
        else:
            story.append(Paragraph(stripped_line, styles['BodyText']))

    try:
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        log.info("Professional PDF generated successfully using Platypus")
        return pdf_bytes
    except Exception as e:
        log.error("Professional PDF (Platypus) generation failed. Falling back to basic.", error=str(e), exc_info=True)
        st.warning(f"Advanced PDF generation failed: {e}. Generating a simpler PDF.", icon="⚠️")
        return text_to_pdf_bytes_basic(resume_text) # Fallback to basic
    finally:
        buffer.close()

# --- Core Analysis Logic ---
@st.cache_data(show_spinner=False)
def run_full_analysis(_resume_bytes, _job_bytes, resume_name, job_name, model_choice, target_role, language):
    analysis_results = {} # Initialize dict for results
    analysis_results['resume_name'] = resume_name
    analysis_results['job_name'] = job_name
    analysis_results['model_choice'] = model_choice
    analysis_results['target_role'] = target_role
    analysis_results['language'] = language

    progress_bar_text_area = st.empty()
    progress_bar = st.progress(0)
    total_steps = 7 # Adjusted total steps

    def update_progress(step_num, text):
        progress_bar_text_area.text(f"⏳ {text}")
        progress_bar.progress(step_num / total_steps)

    try:
        update_progress(0, "Booting up analysis engine...")

        update_progress(1, "📄 Extracting text from your resume...")
        resume_text = extract_text_from_pdf(io.BytesIO(_resume_bytes))
        analysis_results['resume_text'] = resume_text
        log.info("Resume text extracted", length=len(resume_text))

        update_progress(2, "📋 Processing the job description...")
        job_file_obj = io.BytesIO(_job_bytes)
        if job_name.lower().endswith('.txt'):
            job_text = job_file_obj.read().decode("utf-8", errors="replace")
        else:
            job_text = extract_text_from_pdf(job_file_obj)
        analysis_results['job_text'] = job_text
        log.info("Job description text extracted", length=len(job_text))

        update_progress(3, "🧠 Building/Loading semantic understanding model...")
        job_hash = hashlib.md5(job_text.encode()).hexdigest()
        sem_matcher = None # Initialize sem_matcher
        if 'cached_job_semantic_matcher' not in st.session_state or \
           st.session_state.get('cached_job_hash_for_matcher') != job_hash or \
           st.session_state.cached_job_semantic_matcher.language != language:
            log.info("Semantic cache miss. Building new job index.", job_hash=job_hash, language=language)
            job_sentences = [line.strip() for line in job_text.splitlines() if len(line.strip()) > 10]
            if not job_sentences:
                st.warning("Could not extract significant sentences from job description for deep semantic analysis.")
            else:
                sem_matcher = create_semantic_matcher(language=language)
                sem_matcher.build_index(job_sentences)
                st.session_state.cached_job_semantic_matcher = sem_matcher
                st.session_state.cached_job_hash_for_matcher = job_hash
        else:
            log.info("Using cached semantic index for job description.", job_hash=job_hash)
            sem_matcher = st.session_state.cached_job_semantic_matcher
        analysis_results['semantic_matcher'] = sem_matcher

        update_progress(4, "🔍 Performing keyword and formatting checks...")
        analysis_results['token_sim_score'] = compute_token_similarity(resume_text, job_text)
        analysis_results['missing_keywords'] = find_missing_keywords(resume_text, job_text)
        analysis_results['basic_date_issues'] = check_date_formatting(resume_text)
        analysis_results['advanced_date_issues'] = advanced_date_formatting_check(resume_text)
        log.info("Basic analyses complete", score=analysis_results['token_sim_score'])

        update_progress(5, "🎯 Finding semantic links between resume and job...")
        if sem_matcher:
            resume_sentences = [line.strip() for line in resume_text.splitlines() if len(line.strip()) > 10]
            if resume_sentences:
                analysis_results['semantic_results'] = sem_matcher.query(resume_sentences, top_k=3) # top_k adjusted
                log.info("Semantic matching complete.")
            else:
                analysis_results['semantic_results'] = {}
                st.warning("Could not extract significant sentences from resume for deep semantic analysis.")
        else:
             analysis_results['semantic_results'] = {}

        update_progress(6, f"🤖 Generating AI feedback with {model_choice}...")
        if not ENHANCED_PROMPT_FILE.exists():
            st.error(f"Critical Error: Prompt file not found at {ENHANCED_PROMPT_FILE}")
            log.critical("Prompt file missing", path=str(ENHANCED_PROMPT_FILE))
            analysis_results['llm_feedback'] = "Error: AI analysis cannot proceed, prompt template is missing."
        else:
            with open(ENHANCED_PROMPT_FILE, "r", encoding="utf-8") as f:
                raw_prompt_template_from_file = f.read()
            analysis_results['llm_feedback'] = analyze_resume(
                resume_text=resume_text,
                job_text=job_text,
                prompt_template_string=raw_prompt_template_from_file,
                model_name=model_choice,
                target_role=target_role,
                language=language
            )
            log.info("LLM feedback generated", model_used=model_choice)

        update_progress(7, "✅ Analysis complete! Results ready.")
        return analysis_results
    except Exception as e:
        log.exception("Core analysis pipeline failed", exc_info=True)
        st.error(f"A critical error occurred during analysis: {e}")
        progress_bar.empty()
        progress_bar_text_area.empty()
        return None # Ensure None is returned on failure
    finally:
        progress_bar.empty() # Always clear progress elements
        progress_bar_text_area.empty()

# --- Streamlit UI Functions ---
def display_login_register():
    st.set_page_config(page_title="Resume Optimizer", layout="wide", initial_sidebar_state="collapsed") # Wider layout
    st.image(LOGO_PATH, width=150) # Display logo if it exists
    st.title("🧾 Resume Optimizer Agent")
    st.markdown("---")
    st.markdown("#### Welcome! Optimize your resume against job descriptions using AI.")

    # Centering the login/register forms
    # Use columns to create a centered effect or just ensure they don't span full width if layout="centered"
    # If layout="wide", they will be on the left. For centering, you might need more HTML/CSS or nested columns.
    # Let's keep it simple with two columns for now.

    login_tab, register_tab = st.tabs(["🔐 Login", "✨ Create Account"])

    with login_tab:
        # st.subheader("Login to Your Account") # Tab title is enough
        with st.form("login_form_main"):
            username = st.text_input("Username", key="login_username_main", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password_main", placeholder="Enter your password")
            login_submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
            if login_submitted:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    log.info("User logged in", username=username)
                    st.success("Logged in successfully!")
                    st.balloons()
                    st.experimental_rerun() # Use experimental_rerun for cleaner transition
                else:
                    st.error("Invalid username or password. Please try again.")

    with register_tab:
        # st.subheader("Create New Account") # Tab title is enough
        with st.form("register_form_main"):
            new_username = st.text_input("Choose a Username", key="register_username_main", placeholder="At least 4 characters")
            new_password = st.text_input("Choose a Password", type="password", key="register_password_main", placeholder="At least 8 characters")
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password_main", placeholder="Re-enter your password")
            register_submitted = st.form_submit_button("Register & Login", use_container_width=True)
            if register_submitted:
                if not new_username or not new_password:
                    st.error("Username and password cannot be empty.")
                elif len(new_username) < 4:
                    st.error("Username must be at least 4 characters long.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters long.")
                else:
                    if register_user(new_username, new_password):
                        st.success(f"Registration successful, {new_username}! You are now logged in.")
                        log.info("New user registered and auto-logged in", username=new_username)
                        st.session_state.authenticated = True
                        st.session_state.username = new_username
                        st.balloons()
                        st.experimental_rerun()
                    else:
                        st.error("Username already exists or a database error occurred. Please try a different username or contact support if issues persist.")

def display_past_analyses():
    st.header("📊 Your Past Analyses")
    st.markdown("Review feedback and results from your previous resume optimization sessions.")
    st.markdown("---")
    analyses = get_user_analyses(st.session_state.username)
    if analyses:
        for analysis_row in analyses: # analysis_row is a sqlite3.Row object
            analysis = dict(analysis_row) # Convert to dict for easier access
            expander_title = f"📜 {analysis.get('resume_name', 'N/A')} vs {analysis.get('job_name', 'N/A')} ({analysis.get('analysis_date', 'N/A')[:10]})"
            with st.expander(expander_title):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Target Role:** {analysis.get('target_role') if analysis.get('target_role') else 'N/A'}")
                    st.caption(f"**Analysis ID:** {analysis.get('id', 'N/A')}")
                with col2:
                    st.info(f"**Model Used:** {analysis.get('model_used', 'N/A')}")
                    st.caption(f"**Language:** {analysis.get('language', 'N/A')}")

                st.markdown("**AI Feedback Summary:**")
                feedback_summary = analysis.get('feedback', '')
                if len(feedback_summary) > 400:
                    feedback_summary = feedback_summary[:400] + "..."
                st.markdown(f"<div style='padding: 10px; border-left: 3px solid #ccc; background-color: #f9f9f9;'><i>{feedback_summary}</i></div>", unsafe_allow_html=True)
    else:
        st.info("No past analyses found. Start by creating your first analysis! 🚀")

def display_analysis_results(results): # results is st.session_state.analysis_results
    if not results:
        st.warning("No analysis results to display.", icon="⚠️")
        return

    tab_titles = [
        "📊 Summary & Scores",
        "🔑 Keyword Analysis",
        "📅 Date & Formatting",
        "🔗 Semantic Relevance",
        "💡 AI Feedback & Suggestions"
    ]
    tab_summary, tab_keywords, tab_formatting, tab_semantic, tab_ai_feedback = st.tabs(tab_titles)

    with tab_summary:
        st.subheader("Overall Resume Snapshot")
        col1, col2, col3 = st.columns(3)
        sim_score = results.get('token_sim_score', 0.0)
        missing_kw_list = results.get('missing_keywords', [])
        missing_kw_count = len(missing_kw_list)
        date_issues_count = len(results.get('basic_date_issues', [])) + len(results.get('advanced_date_issues', []))

        col1.metric("Similarity Score", f"{sim_score:.1%}", help="Overlap with job description. Higher is generally better.")
        col2.metric("Missing Keywords", missing_kw_count, delta=f"-{missing_kw_count}" if missing_kw_count > 0 else None, delta_color="inverse" if missing_kw_count > 0 else "normal", help="Keywords from job description not found in resume.")
        col3.metric("Date Issues", date_issues_count, delta=f"-{date_issues_count}" if date_issues_count > 0 else None, delta_color="inverse" if date_issues_count > 0 else "normal", help="Potential inconsistencies in date formatting.")
        st.divider()

        st.subheader("🗣️ Action Verb Check")
        action_verb_suggestions = suggest_action_verbs(results.get('resume_text', ''))
        if action_verb_suggestions:
            st.warning("Consider Enhancing Your Action Verbs:")
            for suggestion in action_verb_suggestions:
                st.markdown(f"💡 _{suggestion}_")
        else:
            st.success("✅ Your action verbs appear strong and impactful!")

    with tab_keywords:
        st.subheader("🔑 Keyword Gap Analysis")
        missing_keywords = results.get('missing_keywords', [])
        MAX_KEYWORDS_TO_DISPLAY_COLUMNS = 12
        MAX_KEYWORDS_TO_HIGHLIGHT = 7

        if missing_keywords:
            st.error(f"**{len(missing_keywords)} important keywords/phrases from the job description seem to be missing or underemphasized.**")
            st.markdown(f"Top {min(len(missing_keywords), MAX_KEYWORDS_TO_DISPLAY_COLUMNS)} are listed below. Consider incorporating them naturally where relevant:")

            if len(missing_keywords) <= 6:
                 kw_cols = st.columns(min(len(missing_keywords), 3))
                 for i, keyword in enumerate(missing_keywords[:MAX_KEYWORDS_TO_DISPLAY_COLUMNS]):
                      kw_cols[i % len(kw_cols)].info(f"`{keyword}`")
            else:
                for i in range(0, min(len(missing_keywords), MAX_KEYWORDS_TO_DISPLAY_COLUMNS), 3): # Display in rows of 3
                    kw_chunk = missing_keywords[i:i+3]
                    kw_cols = st.columns(len(kw_chunk))
                    for idx, keyword in enumerate(kw_chunk):
                        kw_cols[idx].info(f"`{keyword}`")

            if len(missing_keywords) > MAX_KEYWORDS_TO_DISPLAY_COLUMNS:
                with st.expander(f"View all {len(missing_keywords)} missing keywords..."):
                    st.caption(", ".join(missing_keywords))
            st.divider()
            st.subheader("📝 Resume with Missing Keywords Highlighted")
            st.caption(f"(Highlights up to {MAX_KEYWORDS_TO_HIGHLIGHT} missing keywords. Focus on natural integration.)")
            highlighted_resume = highlight_missing_keywords(results.get('resume_text', ''), missing_keywords[:MAX_KEYWORDS_TO_HIGHLIGHT])
            st.markdown(f'<div style="border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; background-color: #fafafa; max-height: 400px; overflow-y: auto;">{highlighted_resume}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ Excellent! Your resume appears to cover the key terms from the job description effectively.")

    with tab_formatting:
        st.subheader("📅 Date & Formatting Consistency")
        basic_issues = results.get('basic_date_issues', [])
        advanced_issues = results.get('advanced_date_issues', [])
        if basic_issues or advanced_issues:
            if basic_issues:
                st.error("🚨 Basic Date Formatting Issues Noted:")
                for line in basic_issues:
                    st.markdown(f"   ❌ `{line}` (Tip: Ensure all years are 4-digit, e.g., YYYY).")
            if advanced_issues:
                st.warning("💡 Advanced Date Formatting Suggestions:")
                for issue in advanced_issues:
                    st.markdown(f"   🤔 {issue}")
        else:
            st.success("✅ Your date formatting looks consistent and clear!")

    with tab_semantic:
        st.subheader("🔗 Semantic Relevance to Job Description")
        st.markdown("How well do your resume statements align with the job's requirements, even without exact keywords?")
        semantic_results = results.get('semantic_results', {})
        MAX_SEMANTIC_MATCHES_TO_DISPLAY = 5

        if semantic_results:
            displayed_matches_count = 0
            for res_line, matches_list in list(semantic_results.items()): # Iterate over items
                strong_matches_for_line = [m for m in matches_list if m[1] > 0.60] # Filter for stronger matches
                if not strong_matches_for_line or displayed_matches_count >= MAX_SEMANTIC_MATCHES_TO_DISPLAY:
                    continue

                displayed_matches_count += 1
                expander_title = res_line[:65] + "..." if len(res_line) > 65 else res_line
                with st.expander(f"📌 **Resume Snippet:** _{expander_title}_"):
                    st.markdown(f"**Your Statement:**\n> {res_line}")
                    st.markdown("**Potential Alignments in Job Description:**")
                    for job_req, score in strong_matches_for_line:
                        color = "#28a745" if score > 0.75 else ("#ffc107" if score > 0.65 else "#6c757d")
                        score_percent = f"{score*100:.0f}%"
                        st.markdown(f'<div style="margin-bottom: 8px; padding: 10px; border-left: 4px solid {color}; background-color: {color}1A; border-radius: 4px;">'
                                    f'<strong style="color:{color};">Match Strength: {score_percent}</strong><br>{job_req}</div>',
                                    unsafe_allow_html=True)
            if displayed_matches_count == 0 and semantic_results:
                st.info("Some semantic links found, but none met the high display threshold (>60% similarity). The AI Feedback tab may offer more insights.")
            elif displayed_matches_count < len(semantic_results) and displayed_matches_count > 0:
                 st.caption(f"Showing top {displayed_matches_count} strongest semantic links. Additional, weaker links might exist.")
        else:
            st.info("No significant semantic matches found, or this analysis was skipped. Check the AI Feedback for overall alignment.")

    with tab_ai_feedback:
        st.subheader("💡 AI Feedback & Suggestions")
        llm_feedback_content = results.get('llm_feedback', "No AI feedback was generated for this analysis.")
        st.markdown(llm_feedback_content) # Assumes LLM provides markdown
        st.divider()
        st.subheader("✏️ Your Edits & Notes")
        edited_feedback_current_value = results.get('edited_feedback', llm_feedback_content)
        unique_key_edit_feedback = f"edit_feedback_area_{results.get('resume_name', 'res')}_{results.get('job_name', 'job')}_{st.session_state.username}"

        edited_feedback_new = st.text_area(
            "Modify the AI's feedback or add your notes here. This edited version will be used if you save or generate an enhanced resume.",
            value=edited_feedback_current_value,
            height=300,
            key=unique_key_edit_feedback,
            help="Your changes here are temporary until saved or used for generation."
        )
        if edited_feedback_new != edited_feedback_current_value: # Check if text_area content changed
            st.session_state.analysis_results['edited_feedback'] = edited_feedback_new
            # st.info("Your edits to the feedback have been noted.", icon="📝") # Optional feedback

# --- Main Application Flow ---
def main():
    init_session_state() # Initialize session state first
    init_db()            # Then initialize DB

    # Set page config once at the beginning of main app logic if user is not authenticated
    # If authenticated, it will be set again, which is fine.
    if not st.session_state.authenticated:
        display_login_register() # This function now also calls set_page_config
        st.stop()

    # --- Authenticated App UI ---
    st.set_page_config(page_title="Resume Optimizer Dashboard", layout="wide", initial_sidebar_state="expanded")

    with st.sidebar:
        if Path(LOGO_PATH).is_file():
            st.image(LOGO_PATH, width=120)
        else:
            st.markdown("### 🧾 Resume Optimizer") # Fallback if no logo

        st.markdown(f"#### Welcome, {st.session_state.username}!")
        st.divider()

        page = st.radio(
            "Navigation Menu",
            ["🚀 New Analysis", "📊 Past Analyses"],
            key="navigation_radio_main",
            captions=["Analyze a new resume.", "View your saved history."] # Requires Streamlit 1.29+
        )
        st.divider()

        if st.button("Log Out", key="logout_button_main", type="secondary", use_container_width=True):
            log.info("User logged out", username=st.session_state.username)
            if hasattr(llm_engine, 'unload_current_model'): llm_engine.unload_current_model()
            # Clear all session state keys carefully
            keys_to_delete = list(st.session_state.keys())
            for key in keys_to_delete:
                del st.session_state[key]
            st.experimental_rerun() # Rerun to go back to login

        st.divider()
        st.markdown("---") # A more prominent divider
        if st.button("⚠️ Clear My Analysis History", type="secondary", use_container_width=True, help="Permanently deletes all your saved analyses."):
            st.session_state.confirm_delete_history = True # Trigger confirmation

        if st.session_state.get('confirm_delete_history', False):
            st.error("Are you sure you want to delete ALL your past analyses? This action cannot be undone.")
            col_confirm, col_cancel = st.columns(2)
            if col_confirm.button("Yes, Delete All My Data", type="primary", use_container_width=True):
                if delete_user_analyses(st.session_state.username):
                    st.success("Your analysis history has been cleared.")
                    st.session_state.analysis_results = None
                    st.session_state.confirm_delete_history = False
                    if page == "📊 Past Analyses": st.experimental_rerun()
                else:
                    st.error("Could not clear analysis history.")
            if col_cancel.button("Cancel Deletion", use_container_width=True):
                st.session_state.confirm_delete_history = False
                st.experimental_rerun()

        st.caption(f"© {datetime.now().year} Resume Optimizer Agent")


    # --- Page Content ---
    if page == "📊 Past Analyses":
        display_past_analyses()
    elif page == "🚀 New Analysis":
        st.header("🚀 New Resume Analysis")
        st.markdown("Upload your resume and a job description to get AI-powered feedback and optimization suggestions. Results will appear below once analysis is complete.")
        st.divider()

        with st.form(key="analysis_form_main"):
            st.subheader("📄 Upload Documents")
            col_resume, col_job = st.columns(2)
            with col_resume:
                resume_file = st.file_uploader("1. Your Resume (PDF)", type="pdf", key="resume_upload_main_uploader")
            with col_job:
                job_file = st.file_uploader("2. Job Description (PDF or TXT)", type=["pdf", "txt"], key="job_upload_main_uploader")
            st.divider()

            st.subheader("⚙️ Analysis Configuration")
            col_config1, col_config2 = st.columns(2)
            with col_config1:
                target_role_input = st.text_input(
                    "3. Target Job Title/Industry (Optional):",
                    placeholder="e.g., Senior Data Scientist",
                    key="target_role_input_main_field",
                    help="Helps tailor AI feedback and model choice."
                )
                language_input = st.selectbox(
                    "4. Document Language:",
                    options=["English", "French", "Spanish", "German", "Other"],
                    key="language_select_main_field",
                    help="Select the primary language of your documents."
                )
            with col_config2:
                st.write("**🤖 LLM Model Selection**")
                current_target_role_for_rec = target_role_input
                resume_size_for_rec = "medium"
                if resume_file and hasattr(resume_file, 'size'):
                    resume_size_for_rec = "short" if resume_file.size < 50000 else ("long" if resume_file.size > 150000 else "medium")

                recommendations = get_model_recommendations(current_target_role_for_rec, resume_size_for_rec)
                st.info(f"Recommended: `{recommendations.get('primary', DEFAULT_LLM_MODEL)}`\n_Reason: {recommendations.get('reason', '')}_", icon="💡")

                model_options_list = llm_engine.get_available_model_names()
                try:
                    default_model_index = model_options_list.index(recommendations.get('primary'))
                except (ValueError, AttributeError, IndexError):
                    default_model_index = 0 if model_options_list else 0 # Ensure it doesn't crash if list is empty

                model_choice_input = st.selectbox(
                    "5. Choose LLM Model:",
                    options=model_options_list,
                    index=default_model_index,
                    key="model_select_main_field",
                    help="The recommended model is pre-selected."
                )
            st.divider()
            form_submitted_button = st.form_submit_button("✨ Analyze Resume & Get Feedback", type="primary", use_container_width=True)

        if form_submitted_button:
            if resume_file and job_file:
                log.info("Analysis form submitted", user=st.session_state.username, resume=resume_file.name, job=job_file.name)
                with st.spinner(f"🚀 Launching analysis with '{model_choice_input}'... This can take a few moments."):
                    try:
                        llm_engine.get_llm_instance(model_choice_input) # Load/switch model
                        log.info(f"Model {model_choice_input} is ready.")
                    except Exception as e:
                        st.error(f"Failed to prepare AI model {model_choice_input}: {e}")
                        log.critical(f"Model loading failed for {model_choice_input}", error=str(e), exc_info=True)
                        st.stop() # Critical failure

                # Prepare byte objects for caching
                _resume_bytes_val = resume_file.getvalue()
                _job_bytes_val = job_file.getvalue()

                analysis_output = run_full_analysis(
                    _resume_bytes_val,
                    _job_bytes_val,
                    resume_file.name,
                    job_file.name,
                    model_choice_input,
                    target_role_input,
                    language_input
                )
                st.session_state.analysis_results = analysis_output # Update session state
                if analysis_output is None: # run_full_analysis might return None on error
                    st.error("Analysis could not be completed. Please check logs or try again.", icon="🆘")
                else:
                    st.success("Analysis complete! Results are displayed below.", icon="🎉")
            else:
                st.warning("⚠️ Please upload both your resume and the job description to proceed.", icon="❗")

    # Display results if they exist in session state (after form submission or if loaded from history)
    if st.session_state.analysis_results and page == "🚀 New Analysis": # Only show on new analysis page if results are fresh
        st.divider()
        st.header("📈 Analysis Results Breakdown")
        display_analysis_results(st.session_state.analysis_results)

        st.divider()
        st.header("🚀 Next Steps & Output Generation")
        col_save_action, col_generate_action = st.columns(2)
        current_results_for_action = st.session_state.analysis_results # Use current results

        with col_save_action:
            st.subheader("💾 Save This Analysis")
            if st.button("Save to My History", key="save_analysis_button_action", help="Saves the current analysis results and feedback."):
                feedback_to_save_action = current_results_for_action.get('edited_feedback', current_results_for_action.get('llm_feedback', ''))
                # Create a temporary dict for saving to avoid modifying session_state directly with 'llm_feedback_to_save'
                data_to_save = current_results_for_action.copy()
                data_to_save['llm_feedback_to_save'] = feedback_to_save_action

                if save_analysis(st.session_state.username, data_to_save):
                    st.success("Analysis saved to your history!")
                else:
                    st.error("Failed to save analysis.")

        with col_generate_action:
            st.subheader("📄 Generate Enhanced Resume PDF")
            st.markdown("Use the AI feedback to create an improved version of your resume.")
            feedback_for_generation_action = current_results_for_action.get('edited_feedback', current_results_for_action.get('llm_feedback', ''))

            generate_pdf_button_disabled_action = False
            if not feedback_for_generation_action.strip() or \
               feedback_for_generation_action.startswith("Error:") or \
               feedback_for_generation_action.startswith("⚠️"):
                st.caption("AI feedback is missing or contains errors. PDF generation disabled.")
                generate_pdf_button_disabled_action = True

            if st.button("✨ Rewrite Resume & Generate PDF",
                         key="generate_pdf_button_action",
                         type="primary",
                         disabled=generate_pdf_button_disabled_action,
                         help="The AI will rewrite your resume based on feedback and create a new PDF."):
                with st.spinner("⚙️ Applying feedback and crafting your new resume PDF... This is intensive!"):
                    try:
                        updated_resume_text_action = generate_enhanced_resume(
                            original_resume=current_results_for_action.get('resume_text', ''),
                            feedback=feedback_for_generation_action,
                            model_name=current_results_for_action.get('model_choice', DEFAULT_LLM_MODEL),
                            target_role=current_results_for_action.get('target_role', ''),
                            language=current_results_for_action.get('language', 'English')
                        )
                        if updated_resume_text_action.startswith("⚠️ Error"):
                            st.error(f"LLM failed to rewrite resume: {updated_resume_text_action}")
                            log.error("LLM resume rewrite failed", error_detail=updated_resume_text_action)
                        else:
                            st.info("Generating professionally formatted PDF...")
                            pdf_bytes_action = generate_professional_pdf_bytes(
                                updated_resume_text_action,
                                title=f"Enhanced-{current_results_for_action.get('resume_name', 'resume').replace('.pdf','')}"
                            )
                            if pdf_bytes_action:
                                st.download_button(
                                    label="⬇️ Download Enhanced Resume PDF",
                                    data=pdf_bytes_action,
                                    file_name=f"Enhanced_{current_results_for_action.get('resume_name', 'resume').replace('.pdf', '')}.pdf",
                                    mime="application/pdf",
                                )
                                log.info("Enhanced PDF ready for download", user=st.session_state.username)
                            else:
                                st.error("Failed to generate the PDF document from the rewritten text.")
                    except Exception as e:
                        log.exception("PDF generation/rewrite process failed", exc_info=True)
                        st.error(f"An error occurred during PDF generation: {e}")

if __name__ == "__main__":
    main()