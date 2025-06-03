import io
from logging import log
import re
from turtle import st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Dict
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter

from app import text_to_pdf_bytes


def compute_token_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two documents using bag-of-words.
    Returns a float between 0.0 and 1.0 (higher means more overlap).
    """
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    sim_matrix = cosine_similarity(vectors)
    return float(sim_matrix[0][1])

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
    # Common stopwords to filter out
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'our', 'their', 'as', 'if', 'when',
        'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'all',
        'any', 'some', 'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'now', 'here', 'there', 'then', 'once', 'during', 'before',
        'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
        'again', 'further', 'each', 'few', 'more', 'most', 'other', 'such'
    }
    
    resume_tokens = extract_unique_tokens(resume_text)
    job_tokens = extract_unique_tokens(job_text)
    
    # Filter out stopwords and short tokens
    job_tokens = {token for token in job_tokens if token not in stopwords and len(token) > 2}
    missing = job_tokens - resume_tokens
    
    return sorted(missing)

def check_date_formatting(resume_text: str) -> List[str]:
    """
    Identify lines that contain month names but not a four-digit year. 
    Returns lines potentially missing year formatting.
    """
    suspicious_lines = []
    for line in resume_text.splitlines():
        # Look for a month abbreviation (Jan, Feb, etc.)
        if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", line, re.IGNORECASE):
            # If there is no four-digit year, it's flagged
            if not re.search(r"\b(19|20)\d{2}\b", line):
                suspicious_lines.append(line.strip())
    return suspicious_lines

def advanced_date_formatting_check(resume_text: str) -> List[str]:
    """
    Advanced date formatting analysis to detect inconsistencies and suggest improvements.
    Returns a list of suggestions for date formatting improvements.
    """
    suggestions = []
    lines = resume_text.splitlines()
    
    # Different date patterns found in the resume
    date_patterns = {
        'mm/yyyy': re.compile(r'\b\d{1,2}/\d{4}\b'),
        'mm-yyyy': re.compile(r'\b\d{1,2}-\d{4}\b'),
        'month_year': re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', re.IGNORECASE),
        'mon_year': re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b', re.IGNORECASE),
        'yyyy-mm': re.compile(r'\b\d{4}-\d{1,2}\b'),
        'yyyy/mm': re.compile(r'\b\d{4}/\d{1,2}\b')
    }
    
    found_formats = set()
    
    for line in lines:
        for format_name, pattern in date_patterns.items():
            if pattern.search(line):
                found_formats.add(format_name)
    
    # Check for inconsistencies
    if len(found_formats) > 1:
        suggestions.append(f"Inconsistent date formats detected: {', '.join(found_formats)}. Consider using one consistent format throughout.")
    
    # Check for missing end dates
    current_indicators = re.findall(r'\b(present|current|now)\b', resume_text.lower())
    if current_indicators:
        suggestions.append("Consider using 'Present' consistently for current positions instead of variations like 'current' or 'now'.")
    
    # Check for date ranges
    date_ranges = re.findall(r'(\d{1,2}/\d{4})\s*[-–—]\s*(\d{1,2}/\d{4}|present|current)', resume_text.lower())
    if not date_ranges:
        suggestions.append("Consider using clear date ranges (e.g., '01/2020 - 12/2022') for work experience.")
    
    return suggestions

def highlight_missing_keywords(resume_text: str, missing_keywords: List[str]) -> str:
    """
    Highlight missing keywords in the resume text using HTML for Streamlit display.
    Only highlights the first 10 keywords to avoid cluttering.
    """
    if not missing_keywords:
        return resume_text
    
    highlighted_text = resume_text
    
    # Highlight missing keywords in the text (case-insensitive)
    for keyword in missing_keywords[:10]:  # Limit to first 10 to avoid performance issues
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(
            f'<mark style="background-color: #ffeb3b; color: #000; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{keyword}</mark>',
            highlighted_text
        )
    
    # Convert line breaks to HTML
    highlighted_text = highlighted_text.replace('\n', '<br>')
    
    return f'<div style="font-family: monospace; line-height: 1.6; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">{highlighted_text}</div>'

def extract_skills_section(resume_text: str) -> str:
    """
    Extract the skills section from resume text for targeted analysis.
    """
    lines = resume_text.splitlines()
    skills_section = []
    in_skills_section = False
    
    for line in lines:
        line = line.strip()
        if re.match(r'^(skills?|technical skills?|core competencies|technologies).*$', line, re.IGNORECASE):
            in_skills_section = True
            skills_section.append(line)
            continue
        
        if in_skills_section:
            if re.match(r'^[A-Z][A-Z\s&-]+$', line) and len(line) > 3:  # New section header
                break
            skills_section.append(line)
    
    return '\n'.join(skills_section)

def extract_work_experience_section(resume_text: str) -> str:
    """
    Extract work experience section for targeted analysis.
    """
    lines = resume_text.splitlines()
    experience_section = []
    in_experience_section = False
    
    for line in lines:
        line = line.strip()
        if re.match(r'^(work experience|professional experience|experience|employment history).*$', line, re.IGNORECASE):
            in_experience_section = True
            experience_section.append(line)
            continue
        
        if in_experience_section:
            if re.match(r'^[A-Z][A-Z\s&-]+$', line) and len(line) > 3:  # New section header
                break
            experience_section.append(line)
    
    return '\n'.join(experience_section)

def calculate_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
    """
    Calculate the density of specific keywords in the text.
    Returns a dictionary with keyword density percentages.
    """
    if not keywords:
        return {}
    
    text_lower = text.lower()
    total_words = len(text_lower.split())
    
    keyword_density = {}
    for keyword in keywords:
        keyword_count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
        density = (keyword_count / total_words) * 100 if total_words > 0 else 0
        keyword_density[keyword] = round(density, 2)
    
    return keyword_density

def suggest_action_verbs(resume_text: str) -> List[str]:
    """
    Suggest stronger action verbs to replace weak ones in the resume.
    """
    weak_verbs = ['did', 'made', 'got', 'went', 'said', 'had', 'was', 'were', 'worked']
    strong_alternatives = {
        'did': ['executed', 'performed', 'accomplished', 'achieved'],
        'made': ['created', 'developed', 'produced', 'generated'],
        'got': ['obtained', 'acquired', 'secured', 'earned'],
        'went': ['traveled', 'attended', 'visited', 'participated'],
        'said': ['communicated', 'presented', 'articulated', 'conveyed'],
        'had': ['possessed', 'maintained', 'held', 'retained'],
        'was': ['served as', 'acted as', 'functioned as', 'operated as'],
        'were': ['served as', 'acted as', 'functioned as', 'operated as'],
        'worked': ['collaborated', 'operated', 'functioned', 'contributed']
    }
    
    suggestions = []
    text_lower = resume_text.lower()
    
    for weak_verb in weak_verbs:
        if re.search(r'\b' + weak_verb + r'\b', text_lower):
            alternatives = strong_alternatives.get(weak_verb, [])
            if alternatives:
                suggestions.append(f"Replace '{weak_verb}' with stronger alternatives: {', '.join(alternatives)}")
    
    return suggestions

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

    # Define custom styles
    # Main Title (e.g., Candidate Name - often the first line)
    styles.add(ParagraphStyle(name='ResumeTitle',
                              parent=styles['h1'],
                              fontName='Helvetica-Bold',
                              fontSize=18,
                              spaceAfter=0.2*inch,
                              alignment=TA_CENTER))

    # Section Headers (e.g., Experience, Education, Skills)
    styles.add(ParagraphStyle(name='SectionHeader',
                              parent=styles['h2'],
                              fontName='Helvetica-Bold',
                              fontSize=12,
                              spaceBefore=0.2*inch,
                              spaceAfter=0.1*inch,
                              textColor=HexColor('#2E7D32'))) # Example color

    # Body text for bullet points or general text
    styles.add(ParagraphStyle(name='BodyText',
                              parent=styles['Normal'],
                              fontName='Helvetica',
                              fontSize=10,
                              leading=12, # Line spacing
                              spaceAfter=0.05*inch,
                              alignment=TA_LEFT))

    # Bullet points
    styles.add(ParagraphStyle(name='BulletPoint',
                              parent=styles['BodyText'],
                              leftIndent=0.25*inch,
                              bulletIndent=0.1*inch,
                              firstLineIndent=-0.05*inch, # Hanging indent for bullet
                              spaceAfter=0.05*inch))


    story = []

    # Heuristic to identify potential section headers (all caps or common resume headers)
    # This is very basic and can be improved significantly with more sophisticated parsing
    section_keywords = ['SUMMARY', 'PROFILE', 'OBJECTIVE', 'EXPERIENCE', 'EMPLOYMENT',
                        'EDUCATION', 'SKILLS', 'PROJECTS', 'ACTIVITIES',
                        'AWARDS', 'CERTIFICATIONS', 'REFERENCES', 'TECHNICAL SKILLS',
                        'PROFESSIONAL EXPERIENCE', 'WORK EXPERIENCE']

    lines = resume_text.splitlines()

    if not lines:
        story.append(Paragraph("No content provided for the resume.", styles['BodyText']))
        doc.build(story)
        return buffer.getvalue()

    # Attempt to make the first non-empty line the title
    first_content_line_index = 0
    for i, line in enumerate(lines):
        if line.strip():
            story.append(Paragraph(line.strip(), styles['ResumeTitle']))
            first_content_line_index = i + 1
            break
    else: # No content lines found
        story.append(Paragraph("Resume Appears Empty", styles['ResumeTitle']))


    for line in lines[first_content_line_index:]:
        stripped_line = line.strip()
        if not stripped_line:
            # story.append(Spacer(1, 0.1*inch)) # Add small space for empty lines if desired
            continue

        is_section_header = False
        for keyword in section_keywords:
            if stripped_line.upper().startswith(keyword) and len(stripped_line) < 40: # Simple check
                is_section_header = True
                break
        # A common pattern for section headers is all caps
        if not is_section_header and stripped_line.isupper() and len(stripped_line.split()) < 5 and len(stripped_line) > 3:
             is_section_header = True


        if is_section_header:
            story.append(Paragraph(stripped_line, styles['SectionHeader']))
        elif stripped_line.startswith(('-', '•', '*', '>', 'o')): # Basic bullet detection
            # Remove bullet character and use ReportLab's bulletText
            bullet_text = stripped_line[1:].strip()
            story.append(Paragraph(bullet_text, styles['BulletPoint'], bulletText='•'))
        else:
            story.append(Paragraph(stripped_line, styles['BodyText']))

    try:
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        log.info("Professional PDF generated successfully using Platypus")
        return pdf_bytes
    except Exception as e:
        log.error("Professional PDF generation failed with Platypus", error=str(e), exc_info=True)
        st.error(f"Error generating professional PDF: {e}. Falling back to basic PDF.")
        # Fallback to the simpler PDF generation if Platypus fails
        return text_to_pdf_bytes(resume_text) # Your original simple PDF generator
    finally:
        buffer.close()