# app/streamlit_app.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()  # load .env before importing the graph

from src.graph import build_graph
import streamlit as st

st.set_page_config(page_title="Climate in Focus – IPCC AR6 (SYR)")
st.title("Climate in Focus – IPCC AR6 (SYR)")

# Initialize the graph only once per session
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Form to allow submission with Enter
with st.form("qa"):
    query = st.text_input("Ask a question about the report (EN/PT)")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not query or not query.strip():
        st.warning("Please type a question.")
    else:
        try:
            with st.spinner("Querying the IPCC…"):
                result = st.session_state.graph.invoke(
                    {"query": query.strip(), "contexts": [], "answer": {}}
                )

            # Answer (already comes with [p.X] + disclaimer from the pipeline)
            st.markdown(result["answer"]["answer"])

            # Cited excerpts (keeps ALL chunks, as preferred)
            with st.expander("Cited excerpts"):
                for c in result.get("contexts", []) or []:
                    m = c.get("metadata", {}) or {}
                    p = m.get("page", "?")
                    snippet = (c.get("text") or "").strip().replace("\n", " ")
                    if len(snippet) > 600:
                        snippet = snippet[:600] + "…"
                    st.markdown(f"**p.{p}** — {snippet}")

        except Exception as e:
            st.error(f"Failed to execute the graph: {e}")
