import json
import chromadb
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent /"data" /"chroma"
EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent /"data" /"chunks" /"fastapi_chunks_embedded.jsonl"

chroma_client = chromadb.PersistentClient(path=DATA_PATH)
collection = chroma_client.get_or_create_collection(name="data_embeddings")

def load_embedded_chunks(path: Path = EMBEDDINGS_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]
    
def generate_chunk_ids(chunks: list[dict]) -> list[str]:
    counts = {}
    ids = []
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        currentCount = counts.get(source, 0)
        new_id = f"{source}-{currentCount}"
        ids.append(new_id)
        counts[source] = currentCount + 1
    return ids

def extract_page_content(chunks: list[dict]) -> list[str]:
    page_contents = []
    for chunk in chunks:
        page_contents.append(chunk["page_content"])
    return page_contents

def extract_metadata(chunks: list[dict]) -> list[dict]:
    metadatas = []
    for chunk in chunks:
        metadatas.append(chunk["metadata"])
    return metadatas

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
 
if __name__ == "__main__":
    embedded_chunks = load_embedded_chunks()
    load_chunks_into_collection(embedded_chunks)

