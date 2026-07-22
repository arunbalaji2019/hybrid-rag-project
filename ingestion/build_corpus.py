import json
from pathlib import Path
from ingestion.chunk import chunk_file

DOCS_ROOT = Path(__file__).resolve().parent.parent / "data" / "fastapi_docs" / "docs" / "en" / "docs"
CHUNKS_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "fastapi_chunks.jsonl"

# Directories that are community/meta content, not technical documentation,
# per the "Resources" and "About" groups in docs/en/mkdocs.yml's nav.
EXCLUDE_DIRS = {"resources", "about"}

# Root-level files in the same two nav groups, plus real junk (_llm-test.md is
# internal test fixture content for FastAPI's own translation tooling).
EXCLUDE_ROOT_FILES = {
    "fastapi-people.md",
    "help-fastapi.md",
    "contributing.md",
    "translations.md",
    "project-generation.md",
    "external-links.md",
    "newsletter.md",
    "alternatives.md",
    "history-design-future.md",
    "benchmarks.md",
    "management.md",
    "_llm-test.md",
    "translation-banner.md",
}


def list_included_files(docs_root: Path = DOCS_ROOT) -> list[Path]:
    """
    Walks docs_root and returns every .md file that counts as real
    documentation, excluding community/meta pages and known junk.
    """
    included = []
    for path in docs_root.rglob("*.md"):
        relative_parts = path.relative_to(docs_root).parts
        if relative_parts[0] in EXCLUDE_DIRS:
            continue
        if len(relative_parts) == 1 and relative_parts[0] in EXCLUDE_ROOT_FILES:
            continue
        included.append(path)
    return included

def chunk_included_files(docs_root: Path = DOCS_ROOT) -> list:
    """
    Walks docs_root, finds every .md file that counts as real documentation,
    and returns a list of chunks for each file, with each chunk's metadata
    tagged with its source file path (relative to docs_root).
    """
    included_files = list_included_files(docs_root)
    all_chunks = []
    for file_path in included_files:
        chunks = chunk_file(file_path)
        source = str(file_path.relative_to(docs_root))
        for chunk in chunks:
            chunk.metadata["source"] = source
        all_chunks.extend(chunks)
    return all_chunks


def save_chunks(chunks: list, output_path: Path = CHUNKS_OUTPUT_PATH) -> None:
    """
    Writes chunks to output_path as JSONL - one JSON object per line,
    each containing the chunk's text and metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for chunk in chunks:
            record = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    chunks = chunk_included_files()
    save_chunks(chunks)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_OUTPUT_PATH}")
