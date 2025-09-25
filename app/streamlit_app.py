import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.graph import build_graph

st.set_page_config(
    page_title="Clima em Foco – IPCC AR6 (SYR)",
    page_icon="🌍",
    layout="wide",
)

# ---------- Estilo ----------
st.markdown("""
<style>
:root {
  --ipcc-green: #2E7D32;
  --ipcc-blue: #1565C0;
  --soft-bg: #f6faf7;
}
.block-container { max-width: 980px; padding-top: 1.5rem; }

.hero {
  background: linear-gradient(135deg, rgba(21,101,192,.08), rgba(46,125,50,.10));
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 20px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.hero h1 { margin: 0; font-weight: 800; letter-spacing: .2px; }
.hero .subtitle { margin-top: 4px; color: #5e6a73; }

.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,.08);
  background: #fff;
  font-size: .84rem;
  color: #3a3a3a;
}

.bubble {
  border: 1px solid rgba(0,0,0,.07);
  border-radius: 14px;
  padding: .80rem .95rem;
  line-height: 1.6;
  font-size: 1.02rem;
}
.bubble.user { background: #ffffff; }
.bubble.assistant { background: #fbfbfc; }

.cite-wrap { margin-top: .4rem; }
.cite-card {
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 14px;
  padding: .65rem .8rem;
  margin-bottom: .5rem;
  background: #fff;
}
.cite-title {
  display: inline-flex; align-items: center; gap:6px;
  font-weight: 650; color: #29434e;
}
.page-chip {
  display:inline-block; padding:1px 8px; border-radius:999px;
  background: rgba(21,101,192,.08); color:#0b3a7a; font-size:.82rem;
  border: 1px solid rgba(21,101,192,.15);
}

.sidebar-note { color:#6f6f6f; font-size:.92rem; }
button[kind="secondary"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Configurações")

    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()

    def _clear():
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Olá! Pergunte sobre o IPCC AR6 (SYR). Eu responderei **com citações** no formato `[p.X]`.",
            "contexts": None,
        }]
    st.button("🧹 Limpar chat", type="secondary", use_container_width=True, on_click=_clear)

    st.divider()
    if os.getenv("GOOGLE_API_KEY"):
        model_label = f"Gemini: {os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')}"
    else:
        model_label = f"Ollama: {os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')}"
    st.caption(f"**Modelo:** {model_label}")
    st.caption("**Top-K (vetor):** " + os.getenv("TOP_K", "4"))
    st.caption("**Reordenador (rerank):** " + ("Ligado" if os.getenv("RERANK_ENABLE","1")=="1" else "Desligado"))
    st.caption("**Corpus:** IPCC AR6 – Synthesis Report (Longer Report)")
    st.markdown('<p class="sidebar-note">• As respostas usam apenas trechos do relatório.<br>• Cada frase factual termina com a página citada.</p>', unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <span class="badge">🌍 Clima • Evidências</span>
  <h1>Clima em Foco – IPCC AR6 (SYR)</h1>
  <div class="subtitle">Faça perguntas sobre o relatório. Receba respostas sintéticas, com números e <i>calibrated confidence</i>, sempre citando as páginas.</div>
</div>
""", unsafe_allow_html=True)

# ---------- Estado inicial ----------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Pergunte sobre o IPCC AR6 (SYR).",
        "contexts": None,
    }]

graph = st.session_state.graph

# ---------- Histórico ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        role_class = "assistant" if m["role"] == "assistant" else "user"
        st.markdown(f'<div class="bubble {role_class}">{m["content"]}</div>', unsafe_allow_html=True)

        if m.get("contexts"):
            with st.expander("Trechos citados", expanded=False):
                st.markdown('<div class="cite-wrap">', unsafe_allow_html=True)
                for c in (m["contexts"] or []):
                    meta = c.get("metadata") or {}
                    page = meta.get("page", "?")
                    snippet = (c.get("text") or "").strip().replace("\n", " ")
                    if len(snippet) > 700:
                        snippet = snippet[:700] + "…"
                    st.markdown(
                        f'<div class="cite-card"><span class="cite-title"><span class="page-chip">p.{page}</span> Trecho</span> — {snippet}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

# ---------- Entrada ----------
user_query = st.chat_input("Digite sua pergunta sobre o relatório")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query, "contexts": None})
    with st.chat_message("user"):
        st.markdown(f'<div class="bubble user">{user_query}</div>', unsafe_allow_html=True)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Analisando trechos…"):
                result = graph.invoke({
                    "query": user_query.strip(),
                    "contexts": [],
                    "answer": {},
                    "nonce": time.time(),
                })

                answer_text = (result.get("answer") or {}).get("answer", "").strip() or "_(sem resposta)_"
                contexts = result.get("contexts", [])

                st.markdown(f'<div class="bubble assistant">{answer_text}</div>', unsafe_allow_html=True)

                if contexts:
                    with st.expander("Trechos citados", expanded=False):
                        st.markdown('<div class="cite-wrap">', unsafe_allow_html=True)
                        for c in (contexts or []):
                            meta = c.get("metadata") or {}
                            page = meta.get("page", "?")
                            snippet = (c.get("text") or "").strip().replace("\n", " ")
                            if len(snippet) > 700:
                                snippet = snippet[:700] + "…"
                            st.markdown(
                                f'<div class="cite-card"><span class="cite-title"><span class="page-chip">p.{page}</span> Trecho</span> — {snippet}</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "contexts": contexts,
            })

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Falha ao executar o grafo: {e}")
