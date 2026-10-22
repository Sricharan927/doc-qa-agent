from rag_graph import build_index, build_graph, get_embeddings, get_llm

# Step A: indexing and retrieval
vs, pages, chunks = build_index(["sample.pdf"], get_embeddings())
print("Pages:", pages, "Chunks:", chunks)

results = vs.similarity_search("a topic from your pdf", k=2)   # <- change this
for r in results:
    print("---", r.page_content[:200])

# Step B: the full LangGraph workflow
graph = build_graph(get_llm(), vs)
out = graph.invoke({"question": "Who wrote about seed health status of leafy vegetables?"})   # <- change this
print("ANSWER:", out["answer"])
print("Verified:", out["supported"], "Attempts:", out["attempts"])