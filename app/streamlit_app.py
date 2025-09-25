import os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv() 

import streamlit as st
from src.graph import build_graph

st.set_page_config(
    page_title="Climate in Focus – IPCC AR6 (SYR)",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
/* container mais estreito e sem ruído visual */
.block-container { max-width: 960px; padding-top: 2rem; }

/* título enxuto */
h1 { font-weight: 800; letter-spacing: .2px; margin-bottom: .25rem; }
.subtitle { color: #616161; margin-bottom: 1.2rem; }

/* bolhas do chat: look limpo */
.bubble {
  border: 1px solid rgba(0,0,0,.07);
  border-radius: 14px;
  padding: .75rem .9rem;
  line-height: 1.6;
  font-size: 1.02rem;
}
.bubble.user { background: #ffffff; }
.bubble.assistant { background: #fbfbfc; }

/* cards de citação discretos */
.cite-wrap { margin-top: .35rem; }
.cite-card {
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 12px;
  padding: .65rem .8rem;
  margin-bottom: .5rem;
  background: #fff;
}
.cite-title { font-weight: 600; }
.small-note { color: #6f6f6f; font-size: .92rem; }

/* botões e inputs um tico mais refinados */
button[kind="secondary"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()

    def _clear():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! Ask me about the IPCC AR6 (SYR). I’ll answer **with citations** like `[p.X]`.",
                "contexts": None,
            }
        ]
    st.button("🧹 Clear chat", type="secondary", use_container_width=True, on_click=_clear)

    st.divider()
    if os.getenv("GOOGLE_API_KEY"):
        model_label = f"Gemini: {os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')}"
    else:
        model_label = f"Ollama: {os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')}"
    st.caption(f"**Model:** {model_label}")
    st.caption("**Vector top_k:** " + os.getenv("TOP_K", "4"))
    st.caption("**Rerank:** " + ("ON" if os.getenv("RERANK_ENABLE","1")=="1" else "OFF"))
    st.caption("**Corpus:** IPCC AR6 – Synthesis Report (Longer Report)")

st.title("Climate in Focus – IPCC AR6 (SYR)")
st.markdown('<div class="subtitle">Ask questions about the report.', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me about the IPCC AR6 (SYR).",
            "contexts": None,
        }
    ]

graph = st.session_state.graph

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        role_class = "assistant" if m["role"] == "assistant" else "user"
        st.markdown(f'<div class="bubble {role_class}">{m["content"]}</div>', unsafe_allow_html=True)

        if m.get("contexts"):
            with st.expander("Cited excerpts", expanded=False):
                st.markdown('<div class="cite-wrap">', unsafe_allow_html=True)
                for c in (m["contexts"] or []):
                    meta = c.get("metadata") or {}
                    page = meta.get("page", "?")
                    snippet = (c.get("text") or "").strip().replace("\n", " ")
                    if len(snippet) > 700:
                        snippet = snippet[:700] + "…"
                    st.markdown(
                        f'<div class="cite-card"><span class="cite-title">p.{page}</span> — {snippet}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

user_query = st.chat_input("Ask a question about the report")
if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query, "contexts": None}
    )
    with st.chat_message("user"):
        st.markdown(f'<div class="bubble user">{user_query}</div>', unsafe_allow_html=True)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = graph.invoke({
                    "query": user_query.strip(),
                    "contexts": [],
                    "answer": {},
                    "nonce": time.time(),
                })

                answer_text = (result.get("answer") or {}).get("answer", "").strip() or "_(no answer)_"
                contexts = result.get("contexts", [])

                st.markdown(f'<div class="bubble assistant">{answer_text}</div>', unsafe_allow_html=True)

                if contexts:
                    with st.expander("Cited excerpts", expanded=False):
                        st.markdown('<div class="cite-wrap">', unsafe_allow_html=True)
                        for c in (contexts or []):
                            meta = c.get("metadata") or {}
                            page = meta.get("page", "?")
                            snippet = (c.get("text") or "").strip().replace("\n", " ")
                            if len(snippet) > 700:
                                snippet = snippet[:700] + "…"
                            st.markdown(
                                f'<div class="cite-card"><span class="cite-title">p.{page}</span> — {snippet}</div>',
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
            st.error(f"Failed to execute the graph: {e}")
