"""Streamlit UI.  Run:  streamlit run app.py"""
import os
import tempfile

import streamlit as st

from rag_graph import build_graph, build_index, get_embeddings, get_llm

st.set_page_config(page_title="Document Q&A Agent", page_icon="📄")
st.title("📄 Document Q&A Agent")
st.caption("LangGraph workflow: retrieve → generate → verify")

with st.sidebar:
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    if st.button("Build index", disabled=not files):
        paths = []
        for f in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.read())
            tmp.close()
            paths.append(tmp.name)
        with st.spinner("Reading, chunking and embedding..."):
            vs, n_pages, n_chunks = build_index(
                paths, get_embeddings(), chunk_size=chunk_size
            )
        for p in paths:
            os.unlink(p)
        st.session_state.graph = build_graph(get_llm(), vs)
        st.success(f"Indexed {n_pages} pages into {n_chunks} chunks")

question = st.chat_input("Ask a question about your documents")
if question:
    if "graph" not in st.session_state:
        st.warning("Upload a PDF and click 'Build index' first.")
    else:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            out = st.session_state.graph.invoke({"question": question})
        with st.chat_message("assistant"):
            st.write(out["answer"])
            st.caption(
                f"Verified against sources: {'✅' if out['supported'] else '⚠️ not fully verified'}"
                f" · attempts: {out['attempts']}"
            )
            with st.expander("Retrieved passages"):
                for d in out["docs"]:
                    st.write(d.page_content)
                    st.divider()
