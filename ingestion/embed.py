import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # Load environment variables from .env file

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks" / "fastapi_chunks.jsonl"
client = OpenAI()  # Initialize the OpenAI client


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    """
    Reads the JSONL file written by build_corpus.py back into a list of
    chunk dicts, each with "page_content" and "metadata".
    """
    with path.open() as f:
        return [json.loads(line) for line in f]


def batch_chunks(chunks: list, batch_size: int = 100):
    """
    Yields the chunks in consecutive batches of batch_size.
    """
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embeds every chunk's text via the OpenAI API, in batches, and attaches
    each resulting vector onto its corresponding chunk as "embedding".
    Returns the full list of chunks, each now carrying an embedding.
    """
    embedded_chunks = []
    for batch in batch_chunks(chunks, 100):
        texts = [chunk["page_content"] for chunk in batch]
        response = client.embeddings.create(model="text-embedding-3-small", input=texts)
        for chunk, data in zip(batch, response.data):
            chunk["embedding"] = data.embedding
        embedded_chunks.extend(batch)
    return embedded_chunks


if __name__ == "__main__":
    chunks = load_chunks()
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks. First embedding has {len(embedded_chunks[0]['embedding'])} dimensions.")

