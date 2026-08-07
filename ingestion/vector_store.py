import chromadb
from pathlib import Path

from dotenv import load_dotenv
from langfuse import observe
from openai import OpenAI

from ingestion.chunk_data import (
    load_embedded_chunks,
    extract_page_content,
    extract_metadata,
    generate_chunk_ids,
)

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent /"data" /"chroma"
EMBEDDING_MODEL = "text-embedding-3-small"

chroma_client = chromadb.PersistentClient(path=DATA_PATH)
collection = chroma_client.get_or_create_collection(name="data_embeddings")
openai_client = OpenAI()

def extract_embeddings(chunks: list[dict]) -> list[list[float]]:
    embeddings = []
    for chunk in chunks:
        embeddings.append(chunk["embedding"])
    return embeddings

def load_chunks_into_collection(chunks: list[dict]):
    ids = generate_chunk_ids(chunks)
    documents = extract_page_content(chunks)
    metadatas = extract_metadata(chunks)
    embeddings = extract_embeddings(chunks)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )


@observe(as_type="retriever", name="vector_search")
def search_vector_store(query: str, n: int = 5) -> list[tuple[str, dict]]:
    """
    Embeds query with the same model used at ingestion time, and returns
    the top n (id, chunk) pairs from Chroma, ranked closest-first.
    """
    query_embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=query
    ).data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=n)
    ids = results["ids"][0]
    chunks = [
        {"page_content": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return list(zip(ids, chunks))

if __name__ == "__main__":
    embedded_chunks = load_embedded_chunks()
    load_chunks_into_collection(embedded_chunks)

