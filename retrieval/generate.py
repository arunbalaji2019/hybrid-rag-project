from dotenv import load_dotenv
from langfuse import Langfuse, observe
from openai import OpenAI

load_dotenv()

# Created explicitly (not the lazy default client) with a longer timeout -
# the default was too short for this pipeline. Cross-encoder inference in
# rerank() is CPU-heavy and blocks for a stretch, which was starving
# Langfuse's background span export of time to complete before it gave up.
# Must happen before any @observe-decorated function in this pipeline runs,
# since this becomes the singleton client they all use.
langfuse_client = Langfuse(timeout=60)

GENERATION_MODEL = "gpt-4o-mini"
client = OpenAI()

SYSTEM_PROMPT = (
    "You are an expert assistant answering questions about FastAPI, using only "
    "the documentation excerpts provided as context. If the context doesn't "
    "contain enough information to answer confidently, say so plainly instead "
    "of guessing. Cite which source file(s) you drew from when relevant."
)


def build_context(chunks: list[tuple[str, dict, float]]) -> str:
    """
    Formats reranked (id, chunk, score) results into a text block for the
    prompt - each excerpt labeled with its source file, so the model can
    cite where an answer came from.
    """
    sections = []
    for chunk_id, chunk, _ in chunks:
        source = chunk["metadata"].get("source", "unknown")
        sections.append(f"[Source: {source}]\n{chunk['page_content']}")
    return "\n\n---\n\n".join(sections)


@observe(as_type="generation", name="generate_answer")
def generate_answer(query: str, chunks: list[tuple[str, dict, float]]) -> str:
    """
    Calls the LLM with the reranked context and the user's query, and
    returns the generated answer text.
    """
    context = build_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


@observe(name="rag_pipeline")
def answer_question(query: str) -> str:
    """
    The full pipeline, end to end, as one traced unit: hybrid retrieval ->
    rerank -> generation. Since this calls the other @observe-decorated
    functions internally, they nest under this one trace in Langfuse,
    instead of showing up as separate disconnected traces per question.
    """
    from retrieval.fusion import hybrid_search
    from retrieval.rerank import rerank

    fused = hybrid_search(query, n_per_system=10, n_final=10)
    reranked = rerank(query, fused, top_k=5)
    return generate_answer(query, reranked)


if __name__ == "__main__":
    query = "How do I use response_model_exclude_unset?"
    answer = answer_question(query)

    print("Query:", query)
    print()
    print("Answer:", answer)

    langfuse_client.flush()
