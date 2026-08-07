from langfuse import observe

from ingestion.bm25_store import build_bm25_index, search_bm25
from ingestion.chunk_data import load_embedded_chunks, generate_chunk_ids
from ingestion.vector_store import search_vector_store

# Built once at module load, not per query - BM25 indexing is cheap/local,
# but no reason to redo it on every hybrid_search() call in the same run.
_chunks = load_embedded_chunks()
_ids = generate_chunk_ids(_chunks)
_bm25 = build_bm25_index(_chunks)


@observe(name="reciprocal_rank_fusion")
def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, dict]], k: int = 60
) -> list[tuple[str, dict, float]]:
    """
    Combines multiple ranked (id, chunk) lists into one, using Reciprocal
    Rank Fusion: each chunk's score is the sum of 1/(k + rank) across every
    list it appears in. A chunk ranked highly in multiple lists gets
    boosted; one that only appears in a single list still counts, just
    less. k=60 is the standard damping constant from the original RRF
    paper - it keeps any single very-high rank from dominating the score.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, chunk) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            chunks_by_id[chunk_id] = chunk

    fused_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [(chunk_id, chunks_by_id[chunk_id], scores[chunk_id]) for chunk_id in fused_ids]


@observe(name="hybrid_search")
def hybrid_search(query: str, n_per_system: int = 10, n_final: int = 5) -> list[tuple[str, dict, float]]:
    """
    Runs query through both BM25 and the vector store, fuses the two
    ranked lists with RRF, and returns the top n_final (id, chunk, score).
    """
    bm25_results = search_bm25(_bm25, _ids, _chunks, query, n=n_per_system)
    vector_results = search_vector_store(query, n=n_per_system)
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    return fused[:n_final]


if __name__ == "__main__":
    query = "response_model_exclude_unset"
    for i, (chunk_id, chunk, score) in enumerate(hybrid_search(query, n_final=5)):
        print(f"--- result {i} (id={chunk_id}, score={score:.4f}) ---")
        print("source:", chunk["metadata"].get("source"))
        print(chunk["page_content"][:150])
        print()
