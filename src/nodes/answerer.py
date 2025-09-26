import os
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

from typing import List, Dict
import os, re, textwrap
from dotenv import load_dotenv
load_dotenv()

from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def make_llm():
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        gem_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        return ChatGoogleGenerativeAI(
            model=gem_model,
            temperature=0.0,
        )
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    return ChatOllama(
        model=ollama_model,
        temperature=0.0,
        num_ctx=2048,
        num_predict=256,
        keep_alive="30m",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )

llm = make_llm()
print("LLM ativo:", type(llm).__name__)

FALLBACK = "Não encontrei evidências suficientes no IPCC para responder com confiança."

SYSTEM_PROMPT = """Responda usando APENAS os trechos fornecidos do IPCC AR6 Synthesis Report – Longer Report (SYR).

Regras:
- Escreva em português do Brasil. Mantenha termos técnicos do IPCC como aparecem nos trechos quando não houver tradução inequívoca.
- Reproduza números, unidades, intervalos, rótulos de cenários (ex.: SSP1-2.6) e termos calibrados de confiança exatamente como nos trechos.
- Mencione cenários/regiões/janelas de tempo SOMENTE se constarem nos trechos.
- Cada frase factual deve terminar com uma citação de página no formato [p.X]; se usar vários trechos, encadeie [p.X][p.Y].
- Use apenas páginas que aparecem nos trechos.
- NÃO copie cabeçalhos de seção/figura/tabela (ex.: “Figure 3.2”). Descreva o conteúdo em texto corrido.
- Seja conciso: parágrafos curtos ou bullets quando ajudar.
- Se os trechos forem insuficientes, responda exatamente:
Não encontrei evidências suficientes no IPCC para responder com confiança.
"""

_STOPWORDS = {
    "the","a","an","and","or","of","on","in","at","to","for","with","by","from",
    "is","are","was","were","be","been","being","that","this","these","those",
    "what","which","who","whom","why","how","when","where","than","as","into",
    "about","over","under","it","its","their","there","we","you","your","our",

    "o","a","os","as","um","uma","uns","umas","de","do","da","dos","das","no","na","nos","nas",
    "em","por","para","com","sem","ao","aos","à","às","e","ou","que","como","quando","onde",
    "qual","quais","porque","sobre","entre","até","desde","após","antes","mais","menos"
}

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")

def _build_context(ctxs: List[Dict]) -> str:
    blocks = []
    for c in ctxs:
        txt = (c.get("text") or c.get("page_content") or "").strip()
        pg  = c.get("page") or (c.get("metadata") or {}).get("page")
        if not txt or not pg:
            continue
        blocks.append(f"[p.{pg}]\n{txt}")
    return "\n\n---\n\n".join(blocks)

def _normalize_citations(text: str) -> str:
    text = re.sub(r"\[\s*p\s*\.?\s*(\d+)\s*\]", r"[p.\1]", text)
    text = re.sub(r"\(\s*p\s*\.?\s*(\d+)\s*\)", r"[p.\1]", text)
    text = re.sub(r"\s+\[p\.(\d+)\]", r" [p.\1]", text)
    return text

def _has_any_citation(text: str) -> bool:
    return bool(re.search(r"\[p\.\d+\]", text))

def _keywords(q: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\-\./]+", q.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]

def _extractive_fallback(query: str, ctxs: List[Dict], max_sents: int = 5, min_sents:int = 2) -> str:
    kws = set(_keywords(query))
    picked: List[str] = []
    seen = set()

    def pick_from_text(txt, pg, require_kws=True):
        nonlocal picked
        for sent in _SENT_SPLIT.split(txt):
            s = sent.strip()
            if not s:
                continue
            lower = s.lower()
            if require_kws and kws and not any(k in lower for k in kws):
                continue
            sig = re.sub(r"\s+", " ", lower)
            if sig in seen:
                continue
            seen.add(sig)
            if not re.search(r"\[p\.\d+\]\s*$", s):
                s = s.rstrip(". ") + f" [p.{pg}]"
            picked.append(s)
            if len(picked) >= max_sents:
                break

    for c in ctxs:
        txt = (c.get("text") or c.get("page_content") or "").strip()
        pg  = c.get("page") or (c.get("metadata") or {}).get("page")
        if txt and pg:
            pick_from_text(txt, pg, require_kws=True)
        if len(picked) >= max_sents:
            break

    if len(picked) < min_sents:
        for c in ctxs:
            txt = (c.get("text") or c.get("page_content") or "").strip()
            pg  = c.get("page") or (c.get("metadata") or {}).get("page")
            if txt and pg:
                pick_from_text(txt, pg, require_kws=False)
            if len(picked) >= max_sents:
                break

    if len(picked) < min_sents:
        return FALLBACK
    return picked[0] if len(picked) == 1 else "\n".join(f"- {s}" for s in picked)

def answer(query: str, ctxs: List[Dict]) -> Dict:
    if not ctxs:
        return {"answer": FALLBACK, "contexts": []}

    context_text = _build_context(ctxs)
    user = textwrap.dedent(f"""
    Pergunta:
    {query}

    Trechos do IPCC (use APENAS o que está abaixo):
    {context_text}

    Lembre-se: termine CADA frase factual com [p.X] (ou múltiplas como [p.X][p.Y]).
    """)

    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
    out = llm.invoke(msgs)
    ans = (out.content or "").strip()
    ans = _normalize_citations(ans)

    if not ans or ans.strip() == FALLBACK or not _has_any_citation(ans):
        ans = _extractive_fallback(query, ctxs)

    ans = re.sub(r"[ \t]+", " ", ans).strip()
    return {"answer": ans, "contexts": ctxs}
