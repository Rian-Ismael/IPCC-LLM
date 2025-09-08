# app/streamlit_app.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()  # carrega .env antes de importar o grafo

from src.graph import build_graph
import streamlit as st

st.set_page_config(page_title="Clima em Foco – IPCC AR6 (SYR)")
st.title("Clima em Foco – IPCC AR6 (SYR)")

# Inicializa o grafo uma única vez por sessão
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Form para permitir envio com Enter
with st.form("qa"):
    query = st.text_input("Faça uma pergunta sobre o relatório (PT/EN)")
    submitted = st.form_submit_button("Perguntar")

if submitted:
    if not query or not query.strip():
        st.warning("Digite uma pergunta.")
    else:
        try:
            with st.spinner("Consultando o IPCC…"):
                result = st.session_state.graph.invoke(
                    {"query": query.strip(), "contexts": [], "answer": {}}
                )

            # Resposta (já vem com [p.X] + disclaimer do pipeline)
            st.markdown(result["answer"]["answer"])

            # Trechos citados (mantém TODOS os chunks, conforme você prefere)
            with st.expander("Trechos citados"):
                for c in result.get("contexts", []) or []:
                    m = c.get("metadata", {}) or {}
                    p = m.get("page", "?")
                    snippet = (c.get("text") or "").strip().replace("\n", " ")
                    if len(snippet) > 600:
                        snippet = snippet[:600] + "…"
                    st.markdown(f"**p.{p}** — {snippet}")

        except Exception as e:
            st.error(f"Falha ao executar o grafo: {e}")
