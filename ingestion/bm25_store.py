import re
from rank_bm25 import BM25Okapi
from ingestion.chunk_data import load_embedded_chunks, extract_page_content, generate_chunk_ids

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """
    Minimal BM25 tokenizer: lowercase, extract word sequences (letters,
    digits, underscore). Using \\w+ instead of a plain whitespace split
    matters a lot here - this corpus is full of backtick-wrapped code
    terms like `response_model_exclude_unset`, and a whitespace-only split
    would keep the backticks/punctuation glued to the token, causing exact
    BM25 matches to silently fail. No stemming or stopword removal beyond
    this - a reasonable baseline, not tuned further for now.
    """
    return TOKEN_RE.findall(text.lower())


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Builds a BM25 index from chunks' page_content. Unlike the vector store,
    this has nothing to persist to disk - it's cheap, local, and rebuilt
    from the same chunks file each time it's needed.
    """
    tokenized_corpus = [tokenize(text) for text in extract_page_content(chunks)]
    return BM25Okapi(tokenized_corpus)


def search_bm25(
    bm25: BM25Okapi, ids: list[str], chunks: list[dict], query: str, n: int = 5
) -> list[tuple[str, dict]]:
    """
    Returns the top n (id, chunk) pairs ranked by BM25 score against the
    query - same (id, chunk) shape as vector_store.search_vector_store,
    so results from both can be matched up and fused by ID.
    """
    tokenized_query = tokenize(query)
    items = list(zip(ids, chunks))
    return bm25.get_top_n(tokenized_query, items, n=n)


if __name__ == "__main__":
    chunks = load_embedded_chunks()
    ids = generate_chunk_ids(chunks)
    bm25 = build_bm25_index(chunks)

    # Deliberately an exact-term query - the case hybrid search exists for,
    # where BM25's keyword matching should outperform pure semantic search.
    query = "response_model_exclude_unset"
    results = search_bm25(bm25, ids, chunks, query, n=3)

    for i, (chunk_id, chunk) in enumerate(results):
        print(f"--- result {i} (id={chunk_id}) ---")
        print("source:", chunk["metadata"].get("source"))
        headers = {k: v for k, v in chunk["metadata"].items() if k != "source"}
        print("headers:", headers)
        print(chunk["page_content"][:200])
        print()
