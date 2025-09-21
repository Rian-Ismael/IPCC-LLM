# src/utils/pdf_loader.py
import re
import fitz  # PyMuPDF

# normalizações leves: hífens de quebra, NBSP e ligaduras comuns
_HYPHENS = str.maketrans({
    "\u00ad": "",   # soft hyphen (remove)
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
})
_NBSP = str.maketrans({"\u00a0": " "})
_LIGS = str.maketrans({"\ufb01": "fi", "\ufb02": "fl"})

def normalize_text(raw: str) -> str:
    if not raw:
        return ""
    txt = raw
    # junta palavra-\ncontinuação
    txt = re.sub(r"(\w)[\-­]\s*\n\s*(\w)", r"\1\2", txt)
    # normaliza traços, nbsp e ligaduras
    txt = txt.translate(_HYPHENS).translate(_NBSP).translate(_LIGS)
    # colapsa espaços/linhas
    txt = re.sub(r"[ \t\f\v]+", " ", txt)
    txt = re.sub(r"\s*\n\s*", " ", txt)
    return txt.strip()

def load_pdf_with_metadata(path: str):
    """
    Retorna: [{ 'text': <texto normalizado da página>, 'page': <1-based> }, ...]
    """
    doc = fitz.open(path)
    out = []
    for i in range(len(doc)):
        page = doc[i]
        text = normalize_text(page.get_text("text"))
        out.append({"text": text, "page": i + 1})
    return out
