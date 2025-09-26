# src/utils/pdf_loader.py
import re
import fitz

_HYPHENS = str.maketrans({
    "\u00ad": "",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
})
_NBSP = str.maketrans({"\u00a0": " "})
_LIGS = str.maketrans({"\ufb01": "fi", "\ufb02": "fl"})

def normalize_text(raw: str) -> str:
    if not raw:
        return ""
    txt = raw
    txt = re.sub(r"(\w)[\-­]\s*\n\s*(\w)", r"\1\2", txt)
    txt = txt.translate(_HYPHENS).translate(_NBSP).translate(_LIGS)
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
