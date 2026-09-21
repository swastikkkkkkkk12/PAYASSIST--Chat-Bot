from pathlib import Path

# Project root = one folder above the folder this file lives in
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder that holds all the Markdown knowledge files
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"

# Files placed directly inside knowledge_base (no sub-folder) get this category
DEFAULT_CATEGORY = "general"


def load_documents(base_dir: Path = KNOWLEDGE_BASE) -> list[dict[str, str]]:
    """Read every .md file under base_dir and return them as a list of dicts."""

    # Fail early with a clear message instead of quietly returning nothing
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base folder not found: {base_dir}")

    documents = []

    # sorted() -> files come in the same order on every run
    for file_path in sorted(base_dir.rglob("*.md")):

        # Read the file; skip it (instead of crashing) if it can't be read
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"Skipping {file_path.name}: {error}")
            continue

        # Empty files add nothing, so skip them too
        if not text.strip():
            print(f"Skipping {file_path.name}: file is empty")
            continue

        # Category = first folder inside knowledge_base
        # e.g. knowledge_base/loans/emi/emi_faq.md -> "loans"
        parts = file_path.relative_to(base_dir).parts
        category = parts[0] if len(parts) > 1 else DEFAULT_CATEGORY

        documents.append(
            {
                "text": text,
                "source": file_path.name,
                "category": category,
                "path": str(file_path),
            }
        )

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Total documents loaded: {len(documents)}")

    for doc in documents:
        print(f"- {doc['category']} | {doc['source']}")