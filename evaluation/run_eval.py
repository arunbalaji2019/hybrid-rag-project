import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.fusion import hybrid_search
from retrieval.generate import generate_answer
from retrieval.rerank import rerank

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.jsonl"
JUDGE_MODEL = "gpt-4o-mini"
client = OpenAI()


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def check_retrieval_hit(expected_sources: list[str], reranked_chunks: list[tuple]) -> bool | None:
    """
    True if at least one expected source appears among the final reranked
    chunks' sources. None for entries with no expected_sources (refusal
    cases) - there's no "correct" source to check for those.
    """
    if not expected_sources:
        return None
    retrieved_sources = {chunk["metadata"].get("source") for _, chunk, _ in reranked_chunks}
    return any(source in retrieved_sources for source in expected_sources)


def judge_answer(question: str, rubric: str, answer: str) -> tuple[bool, str]:
    """
    Uses an LLM to grade whether the generated answer satisfies the
    rubric. Returns (passed, reasoning).
    """
    judge_prompt = (
        f"Question: {question}\n\n"
        f"Grading rubric: {rubric}\n\n"
        f"Answer to grade: {answer}\n\n"
        'Does this answer satisfy the rubric? Respond with JSON: '
        '{"passed": true or false, "reasoning": "one or two sentences"}'
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict grader evaluating a RAG system's answers "
                    "against a rubric. Be precise and skeptical - don't pass an "
                    "answer just because it sounds confident."
                ),
            },
            {"role": "user", "content": judge_prompt},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result["passed"], result["reasoning"]


def run_eval() -> list[dict]:
    dataset = load_golden_dataset()
    results = []

    for entry in dataset:
        query = entry["question"]
        fused = hybrid_search(query, n_per_system=10, n_final=10)
        reranked = rerank(query, fused, top_k=5)
        answer = generate_answer(query, reranked)

        retrieval_hit = check_retrieval_hit(entry["expected_sources"], reranked)
        passed, reasoning = judge_answer(query, entry["rubric"], answer)

        results.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "question": query,
                "answer": answer,
                "retrieval_hit": retrieval_hit,
                "judge_passed": passed,
                "judge_reasoning": reasoning,
            }
        )

    return results


def print_summary(results: list[dict]) -> None:
    total = len(results)
    judge_passed = sum(1 for r in results if r["judge_passed"])
    retrieval_checked = [r for r in results if r["retrieval_hit"] is not None]
    retrieval_hits = sum(1 for r in retrieval_checked if r["retrieval_hit"])

    print(f"Judge pass rate: {judge_passed}/{total}")
    if retrieval_checked:
        print(f"Retrieval hit rate: {retrieval_hits}/{len(retrieval_checked)} (entries with an expected source)")
    print()

    for r in results:
        status = "PASS" if r["judge_passed"] else "FAIL"
        if r["retrieval_hit"] is None:
            hit = ""
        else:
            hit = " | retrieval: HIT" if r["retrieval_hit"] else " | retrieval: MISS"
        print(f"[{status}]{hit} ({r['category']}) {r['id']}: {r['question']}")
        if not r["judge_passed"]:
            print(f"    reason: {r['judge_reasoning']}")


if __name__ == "__main__":
    results = run_eval()
    print_summary(results)

    with RESULTS_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nFull results saved to {RESULTS_PATH}")
