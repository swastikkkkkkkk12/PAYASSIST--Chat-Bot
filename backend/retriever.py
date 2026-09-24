"""Find the most relevant knowledge-base chunks for a user question."""

from functools import lru_cache

import numpy as np

from embeddings import create_embeddings, get_model


# Minimum semantic similarity required before a result is considered relevant.
MIN_SCORE = 0.40

# Additional score given when the knowledge-base filename directly matches
# the classified intent.
INTENT_BOOST = 0.15

# Number of cached search combinations.
SEARCH_CACHE_SIZE = 256


@lru_cache(maxsize=1)
def _load_index():
    """Load knowledge-base chunks and their embedding vectors."""
    chunks, vectors = create_embeddings()
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    return chunks, vectors


@lru_cache(maxsize=1)
def _load_model():
    """Load the embedding model once."""
    return get_model()


def warm_up() -> None:
    """Load the knowledge base and embedding model."""
    _load_index()
    _load_model()


def _encode_query(query: str) -> np.ndarray:
    """Create a normalized embedding for the query."""
    return np.asarray(
        _load_model().encode(
            query,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


def _top_k_indexes(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indexes of the highest-scoring results, best first."""
    k = min(k, scores.shape[0])

    if k <= 0:
        return np.empty(0, dtype=np.intp)

    best = np.argpartition(scores, -k)[-k:]

    return best[np.argsort(scores[best])[::-1]]


def _source_matches_intent(source: str, intent: str) -> bool:
    """
    Check whether the knowledge-base filename directly matches the intent.

    Example:
        CREATE_COMPLAINT
        create_complaint.md
    """
    if not intent:
        return False

    source_name = source.rsplit("/", 1)[-1]
    source_name = source_name.rsplit(".", 1)[0].lower()

    normalized_intent = intent.strip().lower()

    return source_name == normalized_intent


@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def _cached_search(
    query: str,
    top_k: int,
    intent: str,
) -> tuple[dict, ...]:
    """Perform cached semantic search with intent-aware ranking."""

    chunks, vectors = _load_index()

    # Create query embedding.
    query_vector = _encode_query(query)

    # Calculate cosine similarity because vectors are normalized.
    scores = vectors @ query_vector

    # Only consider semantically relevant chunks.
    candidates = np.nonzero(scores >= MIN_SCORE)[0]

    if candidates.size == 0:
        return ()

    # Create ranking scores.
    ranking_scores = scores[candidates].copy()

    # Give a small boost to the document whose filename matches
    # the already-classified intent.
    for position, index in enumerate(candidates):
        source = chunks[index]["source"]

        if _source_matches_intent(source, intent):
            ranking_scores[position] += INTENT_BOOST

    # Rank candidates using the boosted score.
    ranked_local = _top_k_indexes(
        ranking_scores,
        candidates.size,
    )

    ranked = candidates[ranked_local]

    # Keep only one result per source file.
    unique_results = []
    seen_sources = set()

    for index in ranked:
        source = chunks[index]["source"]

        if source in seen_sources:
            continue

        seen_sources.add(source)

        unique_results.append(
            {
                "score": float(scores[index]),
                "text": chunks[index]["text"],
                "source": source,
                "category": chunks[index]["category"],
            }
        )

        if len(unique_results) >= top_k:
            break

    return tuple(unique_results)


def search(
    query: str,
    top_k: int = 3,
    intent: str = "",
) -> list[dict]:
    """Search the knowledge base using semantic similarity and intent."""

    query = query.strip()
    intent = intent.strip()

    if not query:
        return []

    # Return copies so callers cannot modify cached results.
    return [
        dict(result)
        for result in _cached_search(
            query,
            top_k,
            intent,
        )
    ]


def main():
    """Run an interactive retrieval test."""

    print("Loading model and knowledge base...")
    warm_up()

    while True:
        try:
            question = input("\nAsk PayAssist (blank to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            break

        results = search(question)

        print("\nMost relevant knowledge:\n")

        if not results:
            print("No relevant knowledge found.")
            continue

        for i, result in enumerate(results, start=1):
            print("=" * 50)
            print(f"Result {i}")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Category: {result['category']}")
            print()
            print(result["text"])


if __name__ == "__main__":
    main()