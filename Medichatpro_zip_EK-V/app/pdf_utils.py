import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a Streamlit-uploaded PDF file.
    """
    text = ""
    # Reset pointer (important if file.read() was called before)
    uploaded_file.seek(0)
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text
