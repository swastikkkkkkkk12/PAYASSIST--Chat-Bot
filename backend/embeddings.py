"""Create vector embeddings for PayAssist knowledge-base chunks.

Embeddings are cached on disk, keyed by a hash of the model + chunk texts, so a
re-run only re-encodes when # the knowledge base (or the model) actually changed.
"""

from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from chunker import create_chunks
from document_loader import load_documents

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64  
NORMALIZE = True  
CACHE_FILE = Path(__file__).resolve().parent / ".embedding_cache" / "embeddings.npz"


@lru_cache(maxsize=1)
def get_model():
    """Load the model once per process. Reuse this for query embeddings too."""
    
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def _fingerprint(texts):
    """Hash of model + settings + chunk texts. Changes whenever the vectors would."""
    digest = hashlib.sha256(f"{MODEL_NAME}|{NORMALIZE}".encode())
    for text in texts:
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "little"))  # length prefix avoids boundary clashes
        digest.update(data)
    return digest.hexdigest()


def _load_cache(fingerprint):
    try:
        with np.load(CACHE_FILE) as cache:
            if cache["fingerprint"].item() == fingerprint:
                return cache["vectors"]
    except Exception:  # missing or corrupt cache -> just recompute
        pass
    return None


def _save_cache(fingerprint, vectors):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp.npz")
    np.savez(tmp, vectors=vectors, fingerprint=fingerprint)
    tmp.replace(CACHE_FILE)  # atomic swap: a crash can't leave a half-written cache


def create_embeddings(use_cache: bool = True):
    """Return (chunks, vectors). Pass use_cache=False to force a re-embed."""
    documents = load_documents()
    chunks = create_chunks(documents)
    if not chunks:
        raise ValueError("No chunks were created - check the knowledge-base documents.")

    texts = [chunk["text"] for chunk in chunks]
    fingerprint = _fingerprint(texts)

    if use_cache:
        vectors = _load_cache(fingerprint)
        if vectors is not None:
            return chunks, vectors

    vectors = get_model().encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE,
    )
    _save_cache(fingerprint, vectors)
    return chunks, vectors


def main():
    parser = argparse.ArgumentParser(description="Embed PayAssist knowledge-base chunks.")
    parser.add_argument("--rebuild", action="store_true", help="ignore the cache and re-embed everything")
    args = parser.parse_args()

    chunks, vectors = create_embeddings(use_cache=not args.rebuild)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Embedding shape: {vectors.shape} ({vectors.dtype})")
    print(f"\nFirst chunk:\n{chunks[0]['text']}")
    print(f"\nFirst embedding (first 5 dims): {vectors[0][:5]}")
    print(f"L2 norm: {np.linalg.norm(vectors[0]):.3f}")


if __name__ == "__main__":
    main()