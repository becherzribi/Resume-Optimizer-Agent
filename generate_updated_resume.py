# generate_updated_resume.py

import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 1. IMPORT THE CORRECT LLM WRAPPER
try:
    from langchain.llms import Ollama
except ModuleNotFoundError:
    from langchain_community.llms import Ollama

def load_resume_text(resume_path: str) -> str:
    """
    Reads a plain‐text version of your resume.
    If you only have a PDF, use parser.py to extract text into a .txt first.
    """
    with open(resume_path, "r", encoding="utf-8") as f:
        return f.read()

def compose_prompt(resume_text: str, feedback_instructions: str) -> str:
    """
    Builds a prompt asking the LLM to rewrite the resume while preserving structure.
    """
    return f"""
You are a professional resume optimization assistant. Rewrite the resume below to incorporate the following changes,
while strictly preserving the original section headings and overall structure. The uploaded resume’s headings are:
(Profile, Education, Work Experience, Skills, etc.). Keep those headings exactly, but update the content under each
heading according to the instructions.

Feedback Instructions:
{feedback_instructions}

Original Resume Text:
{resume_text}

Output the updated resume as plain text, preserving all original headings in the same order.
"""

def rewrite_resume(resume_text: str, feedback_instructions: str, model_name: str = "mistral:latest") -> str:
    """
    Calls the LLM to rewrite the resume text.
    """
    prompt = compose_prompt(resume_text, feedback_instructions)
    llm = Ollama(model=model_name)
    updated_text = llm(prompt)
    return updated_text

def text_to_pdf_bytes(resume_text: str) -> bytes:
    """
    Converts a plain-text resume (with headings) into a PDF and returns the PDF as bytes.
    We use ReportLab to draw each line. If a line would overflow the page height, we add a new page.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Set up a reasonable margin and font
    x_margin = 40
    y_margin = 40
    line_height = 14  # 12pt font + 2pt spacing

    c.setFont("Helvetica", 12)
    # Start at the top of the first page
    y_position = height - y_margin

    for line in resume_text.splitlines():
        # If we run out of vertical space, start a new page
        if y_position < y_margin:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - y_margin

        # Draw the line
        c.drawString(x_margin, y_position, line)
        y_position -= line_height

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_updated_resume_pdf(
    resume_txt_path: str,
    feedback_txt_path: str,
    model_name: str = "mistral:latest",
    output_txt_path: str = "updated_resume.txt",
    output_pdf_path: str = "updated_resume.pdf"
) -> None:
    """
    1. Loads existing resume text (plain .txt).
    2. Reads feedback instructions from a file.
    3. Calls Ollama to rewrite the resume.
    4. Saves the rewritten resume to output_txt_path.
    5. Converts that text into a PDF, saved as output_pdf_path.
    """
    # Step 1: Load existing resume text
    with open(resume_txt_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    # Step 2: Load feedback instructions
    with open(feedback_txt_path, "r", encoding="utf-8") as f:
        feedback_instructions = f.read().strip()

    # Step 3: Rewrite via LLM
    updated_text = rewrite_resume(resume_text, feedback_instructions, model_name)

    # Step 4: Save updated text
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(updated_text)
    print(f"✅ Updated resume text written to: {output_txt_path}")

    # Step 5: Convert to PDF
    pdf_bytes = text_to_pdf_bytes(updated_text)
    with open(output_pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"✅ Updated resume PDF written to: {output_pdf_path}")

if __name__ == "__main__":
    """
    Example usage:
      1) Extract your PDF resume to text first (if you only have PDF):
         from parser import extract_text_from_pdf
         with open("resume.pdf","rb") as pdf_file:
             txt = extract_text_from_pdf(pdf_file)
         with open("my_resume.txt","w", encoding="utf-8") as f:
             f.write(txt)
      2) Write your instructions in feedback.txt
      3) Run:
         python generate_updated_resume.py
    """
    RESUME_TXT = "my_resume.txt"       # Plain-text version of the uploaded CV
    FEEDBACK_TXT = "feedback.txt"      # File containing your bullet-point instructions
    MODEL = "mistral:latest"           # Or llama3.1:latest, deepseek-r1:1.5b

    generate_updated_resume_pdf(
        resume_txt_path=RESUME_TXT,
        feedback_txt_path=FEEDBACK_TXT,
        model_name=MODEL,
        output_txt_path="updated_resume.txt",
        output_pdf_path="updated_resume.pdf"
    )
