"""Measure accuracy on your own test questions.

1. Put your PDF path below and edit tests.json (question + words the correct
   answer must contain).
2. Run:  python -W ignore evaluate.py
3. Change ONE setting (CHUNK_SIZE or BASE_K), run again, and compare.
   Write the real before/after numbers in your README and resume.

The script pauses between questions and retries after rate-limit (429)
errors, so it stays within the free-tier limits of the Gemini API.
"""
import json
import time

from rag_graph import build_graph, build_index, get_embeddings, get_llm

PDF_PATH = "review.pdf"   # <- your document
CHUNK_SIZE = 400          # <- experiment: 400, 800, 1200
BASE_K = 4                # <- experiment: 2, 4, 6
PAUSE_BETWEEN_QUESTIONS = 12   # seconds (free tier: about 15 requests/minute)
RETRY_WAIT = 45                # seconds to wait after a 429 error
MAX_RETRIES = 3


def ask(graph, question):
    """Run one question, waiting and retrying if the free-tier limit is hit."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return graph.invoke({"question": question})
        except Exception as e:
            limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if not limited or attempt == MAX_RETRIES:
                raise
            print(f"  rate limit hit, waiting {RETRY_WAIT}s (retry {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(RETRY_WAIT)


if __name__ == "__main__":
    vs, pages, chunks = build_index([PDF_PATH], get_embeddings(), chunk_size=CHUNK_SIZE)
    print(f"Indexed {pages} pages into {chunks} chunks")
    graph = build_graph(get_llm(), vs, base_k=BASE_K)
    tests = json.load(open("tests.json"))

    correct = 0
    for i, t in enumerate(tests):
        out = ask(graph, t["question"])
        ok = all(w.lower() in out["answer"].lower() for w in t["must_contain"])
        correct += ok
        print(("PASS" if ok else "FAIL"), "-", t["question"])
        if i < len(tests) - 1:
            time.sleep(PAUSE_BETWEEN_QUESTIONS)

    print(f"\nchunk_size={CHUNK_SIZE} k={BASE_K} -> {correct}/{len(tests)} "
          f"= {100 * correct / len(tests):.0f}% accuracy")
