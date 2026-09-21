"""Find the most relevant knowledge-base chunks for a user question."""

from functools import lru_cache

import numpy as np

from embeddings import create_embeddings, get_model


@lru_cache(maxsize=1)
def _load_index():
    """Build chunks + embeddings once per process, not once per question."""
    chunks, vectors = create_embeddings()
    # float32 + contiguous memory -> faster matmul, half the RAM of float64
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return chunks, vectors


@lru_cache(maxsize=1)
def _load_model():
    """Load the embedding model once per process."""
    return get_model()


def warm_up() -> None:
    """Pay the loading cost up front (call this at app startup)."""
    _load_index()
    _load_model()


def _top_k_indexes(scores: np.ndarray, k: int) -> np.ndarray:
    """Indexes of the k best scores, best first, without sorting everything."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.intp)

    best = np.argpartition(scores, -k)[-k:]       # O(n): the k best, unordered
    return best[np.argsort(scores[best])[::-1]]   # sort only those k


def search(query: str, top_k: int = 3) -> list[dict]:
    """Find the top_k most relevant chunks for a user question."""
    chunks, vectors = _load_index()

    query_vector = np.asarray(
        _load_model().encode(query, normalize_embeddings=True),
        dtype=np.float32,
    )

    # Rows are unit-length (normalize_embeddings=True), so dot product == cosine
    scores = vectors @ query_vector

    results = []
    for i in _top_k_indexes(scores, top_k):
        chunk = chunks[i]
        results.append(
            {
                "score": float(scores[i]),
                "text": chunk["text"],
                "source": chunk["source"],
                "category": chunk["category"],
            }
        )
    return results


def main():
    print("Loading model and knowledge base...")
    warm_up()

    while True:
        try:
            question = input("\nAsk PayAssist (blank to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break

        print("\nMost relevant knowledge:\n")
        for i, result in enumerate(search(question), start=1):
            print("=" * 50)
            print(f"Result {i}")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Category: {result['category']}")
            print()
            print(result["text"])


if __name__ == "__main__":
    main()