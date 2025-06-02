# parser.py
import fitz  # PyMuPDF

def extract_text_from_pdf(file_obj) -> str:
    """
    Given a file‐like object from Streamlit's uploader, return its full text.
    """
    # Read the entire PDF into memory as bytes
    pdf_bytes = file_obj.read()
    # Open with PyMuPDF using a bytes stream
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_chunks = []
    for page in doc:
        text_chunks.append(page.get_text())
    # Join pages with newlines
    full_text = "\n".join(text_chunks).strip()
    return full_text
