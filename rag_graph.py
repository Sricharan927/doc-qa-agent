"""Core logic: document indexing + a 3-step LangGraph workflow.

Flow:  retrieve -> generate -> check --(supported?)--> END
                      ^                     |
                      +---- retry (wider) --+   (max 1 retry)
"""
import os
import time
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

load_dotenv()

MAX_ATTEMPTS = 2  # first try + one retry


# ---------- Models ----------
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-flash-latest"), temperature=0
    )


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )


def _text(msg) -> str:
    """Get plain text from a chat model response."""
    return msg.text.strip()


# ---------- Step 1: data preprocessing + indexing ----------
def build_index(pdf_paths: List[str], embeddings, chunk_size=800, chunk_overlap=100,
                batch_size=25, pause=20):
    """Load PDFs, clean text, split into chunks, embed, store in FAISS.

    Chunks are embedded in small batches with a pause between them, so the
    free-tier limit (about 100 embedding requests per minute) is not exceeded.
    """
    pages = []
    for path in pdf_paths:
        pages.extend(PyPDFLoader(path).load())
    for p in pages:  # basic cleaning: collapse whitespace
        p.page_content = " ".join(p.page_content.split())
    pages = [p for p in pages if len(p.page_content) > 30]  # drop empty pages
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)

    vs = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vs is None:
            vs = FAISS.from_documents(batch, embeddings)
        else:
            vs.add_documents(batch)
        if i + batch_size < len(chunks):
            time.sleep(pause)
    return vs, len(pages), len(chunks)


# ---------- Step 2: LangGraph workflow ----------
class State(TypedDict, total=False):
    question: str
    docs: list
    answer: str
    supported: bool
    attempts: int


def build_graph(llm, vectorstore, base_k=4):
    def retrieve(state: State):
        attempts = state.get("attempts", 0)
        k = base_k + 3 * attempts  # retry pulls more context
        docs = vectorstore.similarity_search(state["question"], k=k)
        return {"docs": docs, "attempts": attempts + 1}

    def generate(state: State):
        context = "\n\n".join(
            f"[Page {d.metadata.get('page', '?') + 1 if isinstance(d.metadata.get('page'), int) else '?'}] {d.page_content}"
            for d in state["docs"]
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "Cite page numbers like [Page 3]. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        )
        return {"answer": _text(llm.invoke(prompt))}

    def check(state: State):
        context = "\n\n".join(d.page_content for d in state["docs"])
        prompt = (
            "Is the ANSWER fully supported by the CONTEXT? "
            "Reply with exactly YES or NO.\n\n"
            f"CONTEXT:\n{context}\n\nANSWER:\n{state['answer']}"
        )
        verdict = _text(llm.invoke(prompt)).upper()
        return {"supported": verdict.startswith("YES")}

    def route(state: State):
        if state["supported"] or state["attempts"] >= MAX_ATTEMPTS:
            return END
        return "retrieve"

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("check", check)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", route, {END: END, "retrieve": "retrieve"})
    return g.compile()
