import fitz  # PyMuPDF


def load_pdf_with_metadata(path: str):
    doc = fitz.open(path)
    out = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        out.append({"text": text, "page": page_num + 1})
    return out