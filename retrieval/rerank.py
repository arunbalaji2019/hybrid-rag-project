from langfuse import observe
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = CrossEncoder(RERANK_MODEL)


@observe(name="rerank")
def rerank(
    query: str, candidates: list[tuple[str, dict, float]], top_k: int = 5
) -> list[tuple[str, dict, float]]:
    """
    Re-scores fusion's candidates with a cross-encoder. Unlike the
    embeddings used for initial retrieval (query and document encoded
    *separately*, then compared by cosine similarity - a "bi-encoder"),
    a cross-encoder sees the query and document *together* as one input
    and directly scores relevance - more accurate, but too slow to run
    against the whole corpus. That's why it only runs here, on fusion's
    already-narrowed candidate set, not on all 1764 chunks.

    Returns the top_k candidates, sorted by cross-encoder score
    (replacing fusion's RRF score, not combined with it).
    """
    pairs = [(query, chunk["page_content"]) for _, chunk, _ in candidates]
    scores = _cross_encoder.predict(pairs)

    reranked = [
        (chunk_id, chunk, float(score))
        for (chunk_id, chunk, _), score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda item: item[2], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    from retrieval.fusion import hybrid_search

    query = "response_model_exclude_unset"
    # Retrieve a wider candidate pool from fusion, then rerank narrows it -
    # that's the whole point, so top_k here should be smaller than n_final.
    fused = hybrid_search(query, n_per_system=10, n_final=10)
    reranked = rerank(query, fused, top_k=3)

    for i, (chunk_id, chunk, score) in enumerate(reranked):
        print(f"--- result {i} (id={chunk_id}, score={score:.4f}) ---")
        print("source:", chunk["metadata"].get("source"))
        print(chunk["page_content"][:150])
        print()
