# pdf_loader.py
import re
import fitz  # PyMuPDF

# mapeia traços/hífens exóticos → "-" (ou remove soft-hyphen), NBSP → " ", corrige ligaduras
_HYPHENS = str.maketrans({
    "\u00ad": "",   # soft hyphen (remove)
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign → hyphen
})
_NBSP = str.maketrans({"\u00a0": " "})        # NBSP → espaço normal
_LIGS = str.maketrans({"\ufb01": "fi", "\ufb02": "fl"})  # ligaduras comuns

def normalize_text(raw: str) -> str:
    if not raw:
        return raw
    txt = raw

    # 1) remove hifenização de quebra de linha: "palavra-\ncontinuação" ou "palavra­\ncontinuação"
    txt = re.sub(r"(\w)[\-­]\s*\n\s*(\w)", r"\1\2", txt)

    # 2) normaliza traços, NBSP e ligaduras
    txt = txt.translate(_HYPHENS).translate(_NBSP).translate(_LIGS)

    # 3) colapsa espaços e quebras de linha
    txt = re.sub(r"[ \t\f\v]+", " ", txt)   # múltiplos espaços → 1
    txt = re.sub(r"\s*\n\s*", " ", txt)     # quebra de linha → espaço

    return txt.strip()

def load_pdf_with_metadata(path: str):
    doc = fitz.open(path)
    out = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = normalize_text(text)                  # <<< NORMALIZA AQUI
        out.append({"text": text, "page": page_num + 1})
    return out
