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

st.markdown("""
<style>
:root{
  --bg:#f6fbf7;       /* fundo claro com leve tom "eco" */
  --fg:#0f172a;       /* texto principal (slate escuro) */
  --muted:#5f6b66;    /* texto secundário */
  --card:#ffffff;
  --border:#e3efe6;   /* borda com tom “eco” */
  --radius:14px;

  --sea:#0ea5e9;      /* azul oceano */
  --leaf:#10b981;     /* verde folha */
}

html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--fg); }
#MainMenu, footer {visibility:hidden;}

.block-container{ max-width: 900px; padding-top: 1.1rem; }

/* ===== Header eco com marca d’água do globo ===== */
.header{
  position: relative;
  padding: 18px 18px 20px;
  border: 1px solid var(--border);
  border-radius: 18px;
  background:
    linear-gradient(135deg, rgba(16,185,129,.10), rgba(14,165,233,.10));
  overflow: hidden;
  margin-bottom: 18px;
}
.header:after{
  content:"";
  position:absolute; inset:auto -40px -60px auto; width:340px; height:340px;
  background-image:url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' fill='none' stroke='%23000000' stroke-opacity='.06' stroke-width='2'>\
<circle cx='100' cy='100' r='95'/>\
<ellipse cx='100' cy='100' rx='95' ry='45'/>\
<ellipse cx='100' cy='100' rx='45' ry='95'/>\
<line x1='5' y1='100' x2='195' y2='100'/>\
<line x1='100' y1='5' x2='100' y2='195'/>\
</svg>");
  background-size:contain; background-repeat:no-repeat; pointer-events:none;
}
.header h1{ margin:0; font-weight:800; letter-spacing:.2px; font-size:1.9rem; }
.header .subtitle{ margin-top:6px; color:var(--muted); font-size:.95rem; }
.badges{ display:flex; gap:8px; margin-bottom:6px; flex-wrap:wrap; }
.badge{
  display:inline-block; padding:2px 10px; border-radius:999px;
  border:1px solid var(--border); background:#fff; color:#245a49;
  font-size:.80rem;
}

/* ===== Bubbles ===== */
.bubble{
  background:var(--card); border:1px solid var(--border);
  border-radius:var(--radius); padding:.9rem 1rem;
  line-height:1.55; font-size:1.02rem;
}
.bubble.user{ border-left:3px solid var(--leaf); }
.bubble.assistant{ border-left:3px solid var(--sea); }

/* ===== Citações ===== */
.cite-wrap{ margin-top:.4rem; }
.cite-card{ border:1px solid var(--border); border-radius:var(--radius);
           padding:.65rem .8rem; margin-bottom:.5rem; background:var(--card); }
.cite-title{ display:inline-flex; align-items:center; gap:6px; font-weight:650; color:#0f172a; }
.page-chip{
  display:inline-block; padding:1px 8px; border-radius:999px;
  background:rgba(14,165,233,.08); color:#074b6a; font-size:.82rem;
  border:1px solid rgba(14,165,233,.20);
}

/* ===== Sidebar minimal ===== */
[data-testid="stSidebar"]{ background:var(--bg); border-right:1px solid var(--border); }
.sidebar-caption{ font-size:.92rem; color:var(--fg); }
.sidebar-muted{ color:var(--muted); }

/* ===== Input pill ===== */
[data-testid="stChatInput"] > div{
  border:1px solid var(--border) !important; border-radius:999px !important;
}
</style>
""", unsafe_allow_html=True)

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

def _clear():
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Pergunte sobre o IPCC AR6 (SYR).",
        "contexts": None,
    }]

with st.sidebar:
    st.markdown("### ⚙️")
    st.button("🧹 Limpar chat", type="secondary", use_container_width=True, on_click=_clear)

    st.divider()
    if os.getenv("GOOGLE_API_KEY"):
        model_label = f"Gemini: {os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')}"
    else:
        model_label = f"Ollama: {os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')}"
    st.caption(f"**Modelo:** {model_label}", help=None)

    rerank_on = (os.getenv("RERANK_ENABLE", "1") == "1")
    st.caption(f"**Rerank:** {'Ligado' if rerank_on else 'Desligado'}")

st.markdown(f"""
<div class="header">
  <div class="badges">
    <span class="badge">🌍 Painel Intergovernamental sobre Mudanças Climáticas (IPCC)</span>
  </div>
  <h1>Clima em Foco – IPCC AR6 (SYR)</h1>
  <div class="subtitle">Pergunte sobre o relatório. Respostas objetivas, com números e citações de páginas.</div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Pergunte sobre o IPCC AR6 (SYR).",
        "contexts": None,
    }]

graph = st.session_state.graph

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        role_class = "assistant" if m["role"] == "assistant" else "user"
        st.markdown(f'<div class="bubble {role_class}">{m["content"]}</div>', unsafe_allow_html=True)

        # Citações (se houver)
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
