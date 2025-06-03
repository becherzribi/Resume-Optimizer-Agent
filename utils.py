# utils.py
import io
import re
# Remove: from logging import log # Not needed, use structlog if necessary
# Remove: from turtle import st # Incorrect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Dict, Tuple # Added Tuple for type hinting

# ReportLab imports for PDF generation (needed for both professional and basic)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas # For the basic PDF generator
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER # TA_JUSTIFY not used in current version
from reportlab.lib.colors import HexColor

# If you intend to use structlog here:
import structlog
log = structlog.get_logger(__name__) # Get a logger specific to this module

# --- Text Analysis Utilities ---

def compute_token_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two documents using bag-of-words.
    Returns a float between 0.0 and 1.0 (higher means more overlap).
    """
    if not text1.strip() or not text2.strip(): # Handle empty inputs
        return 0.0
    try:
        vectorizer = CountVectorizer().fit_transform([text1.lower(), text2.lower()])
        vectors = vectorizer.toarray()
        if vectors.shape[0] < 2 or vectors.shape[1] == 0: # Ensure valid matrix for cosine_similarity
            return 0.0
        sim_matrix = cosine_similarity(vectors)
        return float(sim_matrix[0][1])
    except ValueError: # Handles cases where vocabulary might be empty after processing
        log.warning("ValueError in compute_token_similarity, likely due to empty vocabulary after stopword removal or very short texts.")
        return 0.0


def extract_unique_tokens(text: str) -> Set[str]:
    """
    Lowercase + extract all alphanumeric tokens. Returns a set of unique tokens.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)

def find_missing_keywords(resume_text: str, job_text: str) -> List[str]:
    """
    Returns a sorted list of tokens that appear in the job description
    but do not appear in the resume. Filters out common stopwords.
    """
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'our', 'their', 'am', 'is', 'not', # Added more common ones
        'experience', 'responsibilities', 'requirements', 'skills', 'work', 'job', 'company', 'team', # Domain specific, but often noisy
        'ability', 'knowledge', 'strong', 'excellent', 'good', 'communication', 'etc', 'eg', 'role', 'position',
        'from', 'as', 'if', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'all', 'any', 'some', 'no',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there', 'then', 'once', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'each', 'few', 'more', 'most', 'other', 'such',
        'years', 'year', 'month', 'months' # Time related
    }
    resume_tokens = extract_unique_tokens(resume_text)
    job_tokens = extract_unique_tokens(job_text)

    # Filter out stopwords, short tokens (1-2 chars), and pure numbers
    job_keywords = {
        token for token in job_tokens
        if token not in stopwords and len(token) > 2 and not token.isdigit()
    }
    resume_keywords = {
        token for token in resume_tokens
        if token not in stopwords and len(token) > 2 and not token.isdigit()
    }

    missing = job_keywords - resume_keywords
    return sorted(list(missing))


def check_date_formatting(resume_text: str) -> List[str]:
    """
    Identify lines that contain month names but not a four-digit year.
    Returns lines potentially missing year formatting.
    """
    suspicious_lines = []
    # Regex to find common month abbreviations or full names
    month_pattern = r"\b(Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?|January|February|March|April|June|July|August|September|October|November|December)\b"
    year_pattern = r"\b(19\d{2}|20\d{2})\b" # Matches 19xx or 20xx

    for line in resume_text.splitlines():
        if re.search(month_pattern, line, re.IGNORECASE):
            if not re.search(year_pattern, line):
                suspicious_lines.append(line.strip())
    return suspicious_lines


def advanced_date_formatting_check(resume_text: str) -> List[str]:
    """
    Advanced date formatting analysis to detect inconsistencies and suggest improvements.
    Returns a list of suggestions for date formatting improvements.
    """
    suggestions = []
    lines = resume_text.splitlines()
    date_patterns = {
        'MM/YYYY': re.compile(r'\b(0?[1-9]|1[0-2])/\d{4}\b'),
        'M/YYYY': re.compile(r'\b(0?[1-9]|1[0-2])/\d{4}\b'), # Handles M/YYYY too
        'MM-YYYY': re.compile(r'\b(0?[1-9]|1[0-2])-\d{4}\b'),
        'Month YYYY': re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', re.IGNORECASE),
        'Mon YYYY': re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}\b', re.IGNORECASE), # Added optional period
        'YYYY-MM': re.compile(r'\b\d{4}-(0?[1-9]|1[0-2])\b'),
        'YYYY/MM': re.compile(r'\b\d{4}/(0?[1-9]|1[0-2])\b'),
        'YYYY': re.compile(r'\b(19\d{2}|20\d{2})\b(?!\s*(?:to|-|–|—)\s*(?:Present|Current|19\d{2}|20\d{2}))') # Year only if not part of a range
    }
    found_formats = set()
    date_lines = []

    for line in lines:
        for format_name, pattern in date_patterns.items():
            if pattern.search(line):
                found_formats.add(format_name)
                date_lines.append(line)
                break # Assume one dominant format per line for this check

    if len(found_formats) > 1:
        suggestions.append(f"Inconsistent date formats found: {', '.join(sorted(list(found_formats)))}. Strive for one consistent style (e.g., 'Month YYYY' or 'MM/YYYY').")

    # Check for "Present" consistency
    present_variations = re.findall(r'\b(present|current|now|ongoing|to date)\b', resume_text, re.IGNORECASE)
    if len(set(v.lower() for v in present_variations)) > 1:
        suggestions.append(f"Multiple terms for ongoing roles found (e.g., {', '.join(set(present_variations))}). Standardize to 'Present'.")
    elif not present_variations and "experience" in resume_text.lower(): # If no "present" but has experience
        # This is harder to detect accurately without knowing current role
        pass

    # Check for ambiguous year-only entries if not clearly for graduation
    # This is complex and might produce false positives, so keep it simple or omit
    # for line in date_lines:
    #     if date_patterns['YYYY'].search(line) and not any(kw in line.lower() for kw in ['graduate', 'degree', 'university', 'college']):
    #         if not re.search(r'[-–—to]\s*(Present|Current|Now|Ongoing|To Date|\d{4})', line, re.IGNORECASE): # Not part of a clear range
    #             suggestions.append(f"Ambiguous year-only date found: '{line.strip()}'. Clarify if it's a start/end year or a duration.")

    return suggestions


def highlight_missing_keywords(resume_text: str, missing_keywords: List[str]) -> str:
    """
    Highlight missing keywords in the resume text using HTML.
    """
    if not missing_keywords or not resume_text:
        return resume_text.replace('\n', '<br>') if resume_text else ""

    highlighted_text = resume_text
    # Sort by length descending to match longer phrases first
    sorted_keywords = sorted(missing_keywords, key=len, reverse=True)

    for keyword in sorted_keywords: # Iterate up to MAX_HIGHLIGHT_KEYWORDS (now handled in app.py)
        try:
            # Regex to match whole word/phrase, case-insensitive
            # Using \b for word boundaries, but need to be careful if keyword has punctuation
            # For simplicity, escape the keyword. If keywords are multi-word, this is important.
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<mark style="background-color: #FFF3CD; color: #664D03; padding: 1px 3px; border-radius: 3px; font-weight: 500;">{keyword}</mark>',
                highlighted_text
            )
        except re.error:
            log.warning(f"Regex error highlighting keyword: {keyword}")
            continue # Skip problematic keyword

    return highlighted_text.replace('\n', '<br>')


def extract_skills_section(resume_text: str) -> str:
    """
    Extract the skills section from resume text. (Basic implementation)
    """
    # This is a very basic heuristic and might need to be much more robust
    # based on common resume formats.
    lines = resume_text.splitlines()
    skills_section_lines = []
    in_skills_section = False
    # Common section headers that might precede skills or be skills itself
    section_starters = r'^(skills?|technical skills?|core competencies|technologies|proficiencies|expertise|summary|profile|objective)\b'
    # Common section headers that might follow skills
    section_enders = r'^(experience|employment|education|projects|awards|certifications|publications|references)\b'

    for line in lines:
        line_lower = line.strip().lower()
        if re.match(section_starters, line_lower, re.IGNORECASE):
            in_skills_section = True
            skills_section_lines.append(line.strip()) # Include the header
            continue

        if in_skills_section:
            if re.match(section_enders, line_lower, re.IGNORECASE) and not line_lower.startswith("skills"): # if it's a new section
                break # Stop collecting for skills
            if line.strip(): # Add non-empty lines
                skills_section_lines.append(line.strip())
    return '\n'.join(skills_section_lines) if skills_section_lines else ""


def extract_work_experience_section(resume_text: str) -> str:
    """
    Extract work experience section. (Basic implementation)
    """
    lines = resume_text.splitlines()
    experience_section_lines = []
    in_experience_section = False
    section_starters = r'^(work experience|professional experience|experience|employment history|career history)\b'
    # Consider what typically follows experience: Education, Skills, Projects etc.
    section_enders = r'^(education|skills|technical skills|projects|awards|certifications|publications|references|summary|profile|objective)\b'

    for line in lines:
        line_lower = line.strip().lower()
        if re.match(section_starters, line_lower, re.IGNORECASE):
            in_experience_section = True
            experience_section_lines.append(line.strip()) # Include the header
            continue

        if in_experience_section:
            if re.match(section_enders, line_lower, re.IGNORECASE):
                break
            if line.strip():
                experience_section_lines.append(line.strip())
    return '\n'.join(experience_section_lines) if experience_section_lines else ""


def calculate_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
    """
    Calculate the density of specific keywords in the text.
    Returns a dictionary with keyword density percentages.
    """
    if not keywords or not text.strip():
        return {}

    text_lower = text.lower()
    words_in_text = re.findall(r"\b\w+\b", text_lower)
    total_words = len(words_in_text)
    if total_words == 0:
        return {keyword: 0.0 for keyword in keywords}

    keyword_density_map = {}
    for keyword in keywords:
        kw_lower = keyword.lower()
        # Count occurrences of the keyword as a whole word/phrase
        # For multi-word keywords, this simple findall might not be perfect
        # but it's a common approach.
        # If keyword is "data science", it counts "data science", not "data" and "science" separately here.
        keyword_count = len(re.findall(r'\b' + re.escape(kw_lower) + r'\b', text_lower))
        density = (keyword_count / total_words) * 100
        keyword_density_map[keyword] = round(density, 2)
    return keyword_density_map


def suggest_action_verbs(resume_text: str) -> List[str]:
    """
    Suggest stronger action verbs based on common weak ones.
    """
    # Expanded list of weak verbs and more diverse suggestions
    verb_map = {
        'assisted': ['Supported', 'Facilitated', 'Contributed to', 'Aided'],
        'responsible for': ['Managed', 'Oversaw', 'Directed', 'Led', 'Orchestrated'],
        'worked on': ['Developed', 'Engineered', 'Implemented', 'Executed', 'Contributed to'],
        'did': ['Performed', 'Executed', 'Completed', 'Accomplished'],
        'made': ['Created', 'Developed', 'Designed', 'Produced', 'Generated'],
        'helped': ['Facilitated', 'Supported', 'Enabled', 'Contributed to', 'Streamlined'],
        'handled': ['Managed', 'Processed', 'Administered', 'Coordinated'],
        'participated in': ['Contributed to', 'Engaged in', 'Collaborated on', 'Played a key role in'],
        'used': ['Leveraged', 'Utilized', 'Applied', 'Employed'],
        'showed': ['Demonstrated', 'Presented', 'Illustrated', 'Evidenced'],
        'tasked with': ['Spearheaded', 'Executed tasks for', 'Managed responsibilities for'],
        'involved in': ['Played an integral role in', 'Contributed significantly to', 'Engaged in']
    }
    suggestions_found = []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', resume_text) # Split into sentences

    for sentence in sentences:
        for weak_phrase, strong_options in verb_map.items():
            # Use regex for case-insensitive whole word/phrase matching
            # Ensure weak_phrase is properly escaped for regex
            pattern = r'\b' + re.escape(weak_phrase) + r'\b'
            if re.search(pattern, sentence, re.IGNORECASE):
                # Find the actual match to display in the suggestion
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    actual_weak_phrase = match.group(0)
                    suggestions_found.append(
                        f"In \"_{sentence.strip()[:80]}..._\", consider replacing '`{actual_weak_phrase}`' with: {', '.join(strong_options)}."
                    )
                    break # Move to next sentence once a weak phrase is found in current one
    return suggestions_found


# --- PDF Generation Utilities (Both basic and professional) ---

def text_to_pdf_bytes_basic(resume_text: str) -> bytes:
    """Converts plain text resume to a very basic PDF using ReportLab Canvas. (Fallback)"""
    buffer = io.BytesIO()
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        # Margins
        x_margin = 0.75 * inch
        y_margin = 0.75 * inch
        width, height = letter
        line_height = 12  # For 10pt font
        c.setFont("Helvetica", 10)
        y = height - y_margin

        for line_text in resume_text.splitlines():
            if y < y_margin: # New page if no space
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - y_margin
            c.drawString(x_margin, y, line_text)
            y -= line_height
        c.save()
        pdf_bytes = buffer.getvalue()
        log.debug("Basic PDF generated successfully")
        return pdf_bytes
    except Exception as e:
        log.error("Basic PDF generation failed", error=str(e), exc_info=True)
        # In a util, avoid st.error directly. Let the caller handle UI.
        return b"" # Return empty bytes on failure
    finally:
        buffer.close()


def generate_professional_pdf_bytes(resume_text: str, title: str = "Optimized Resume") -> bytes:
    """
    Generates a more professionally formatted PDF from plain text resume using ReportLab Platypus.
    """
    buffer = io.BytesIO()
    # Increased margins slightly for a less cramped look
    doc = SimpleDocTemplate(buffer,
                            pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch,
                            title=title)
    styles = getSampleStyleSheet()

    # Custom Styles
    styles.add(ParagraphStyle(name='ResumeTitle', parent=styles['h1'], fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, spaceAfter=0.1*inch))
    styles.add(ParagraphStyle(name='ContactInfo', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, spaceAfter=0.2*inch, leading=11))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['h2'], fontName='Helvetica-Bold', fontSize=11, spaceBefore=0.15*inch, spaceAfter=0.05*inch, textColor=HexColor('#333333'), keepWithNext=1, borderPadding=2, leading=14)) # Darker grey
    styles.add(ParagraphStyle(name='BodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=12.5, spaceAfter=0.03*inch, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='BulletPoint', parent=styles['BodyText'], leftIndent=0.25*inch, bulletIndent=0.1*inch, firstLineIndent=-0.1*inch, spaceBefore=0.02*inch, spaceAfter=0.02*inch)) # Tighter bullet spacing

    story = []
    section_keywords = ['SUMMARY', 'PROFILE', 'OBJECTIVE', 'EXPERIENCE', 'EMPLOYMENT', 'EDUCATION', 'SKILLS', 'PROJECTS', 'ACTIVITIES', 'AWARDS', 'CERTIFICATIONS', 'REFERENCES', 'TECHNICAL SKILLS', 'PROFESSIONAL EXPERIENCE', 'WORK EXPERIENCE', 'CONTACT', 'LINKS', 'PUBLICATIONS', 'VOLUNTEER']
    lines = [line for line in resume_text.splitlines() if line.strip()] # Remove empty lines upfront

    if not lines:
        story.append(Paragraph("Resume content is empty.", styles['BodyText']))
        doc.build(story)
        return buffer.getvalue()

    # Title and Contact Info Heuristics
    # Assumes first few lines might be Name, Phone, Email, LinkedIn, Portfolio
    header_lines_count = 0
    for i, line in enumerate(lines):
        if i == 0: # First line is likely the name
            story.append(Paragraph(line.strip(), styles['ResumeTitle']))
            header_lines_count += 1
        elif i < 4 and ("@" in line or "linkedin.com" in line or "github.com" in line or re.search(r'\d', line)): # Contact info
            story.append(Paragraph(line.strip(), styles['ContactInfo']))
            header_lines_count += 1
        else:
            break # End of header section

    # Process remaining lines for sections and content
    current_section_content = []
    for line_idx, line_text in enumerate(lines[header_lines_count:]):
        stripped_line = line_text.strip()
        is_section_header = False
        upper_stripped_line = stripped_line.upper()

        for keyword in section_keywords:
            # Header if it IS a keyword or STARTS with a keyword and is short
            if upper_stripped_line == keyword or (upper_stripped_line.startswith(keyword) and len(stripped_line.split()) <= 3):
                is_section_header = True
                break
        # Also consider all-caps short lines as headers
        if not is_section_header and stripped_line.isupper() and len(stripped_line.split()) < 5 and len(stripped_line) > 2 and len(stripped_line) < 30:
            is_section_header = True

        if is_section_header:
            story.append(Paragraph(stripped_line, styles['SectionHeader']))
        elif stripped_line.startswith(('-', '•', '*', '>', 'o', '▪')) and len(stripped_line) > 1:
            bullet_text = stripped_line[1:].strip()
            if bullet_text:
                story.append(Paragraph(bullet_text, styles['BulletPoint'], bulletText='•'))
        else:
            # Regular body text, could be part of a paragraph under a header or a simple line
            story.append(Paragraph(stripped_line, styles['BodyText']))

    try:
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        log.info("Professional PDF generated successfully using Platypus")
        return pdf_bytes
    except Exception as e:
        log.error("Professional PDF (Platypus) generation failed. Falling back to basic.", error=str(e), exc_info=True)
        # Avoid st.error in utils. Let app.py handle UI.
        # Raising an exception or returning a specific error code might be better than silent fallback.
        # For now, logging and returning basic PDF bytes.
        return text_to_pdf_bytes_basic(resume_text) # Fallback
    finally:
        buffer.close()
