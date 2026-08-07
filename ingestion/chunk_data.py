import json
from pathlib import Path

EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "fastapi_chunks_embedded.jsonl"


def load_embedded_chunks(path: Path = EMBEDDINGS_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def extract_page_content(chunks: list[dict]) -> list[str]:
    return [chunk["page_content"] for chunk in chunks]


def extract_metadata(chunks: list[dict]) -> list[dict]:
    return [chunk["metadata"] for chunk in chunks]


def generate_chunk_ids(chunks: list[dict]) -> list[str]:
    """
    Builds a stable "{source}-{index within that source}" ID per chunk.
    Shared by both retrieval backends (Chroma, BM25) so results from each
    can be matched up and fused by ID later.
    """
    counts = {}
    ids = []
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        current_count = counts.get(source, 0)
        ids.append(f"{source}-{current_count}")
        counts[source] = current_count + 1
    return ids
