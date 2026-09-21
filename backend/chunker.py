"""Split knowledge-base documents into overlapping chunks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_documents

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
METADATA_KEYS = ("source", "category", "path")

# Built once and reused for every document.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def iter_chunks(
    documents: Iterable[dict],
    splitter: RecursiveCharacterTextSplitter = text_splitter,
) -> Iterator[dict]:
    """Yield chunks one at a time, so a big corpus never sits in memory twice."""
    for document in documents:
        metadata = {key: document[key] for key in METADATA_KEYS}

        for index, text in enumerate(splitter.split_text(document["text"])):
            yield {
                "text": text,
                **metadata,
                "chunk_id": index,  # position inside its document (restarts at 0)
                "id": f"{document['path']}::{index}",  # unique across documents
            }


def create_chunks(
    documents: Iterable[dict],
    splitter: RecursiveCharacterTextSplitter = text_splitter,
) -> list[dict]:
    """Return every chunk as a list."""
    return list(iter_chunks(documents, splitter))


def main() -> None:
    documents = load_documents()
    chunks = create_chunks(documents)

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print("\nFirst 3 chunks:\n")

    for chunk in chunks[:3]:
        print("-" * 32)
        print(f"Source: {chunk['source']}")
        print(f"Category: {chunk['category']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(chunk["text"])


if __name__ == "__main__":
    main()