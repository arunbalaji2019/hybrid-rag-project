from retrieval.generate import answer_question, langfuse_client

BANNER = (
    "FastAPI Docs Assistant - hybrid RAG (BM25 + vector search + rerank)\n"
    "Ask a question about FastAPI, or type 'exit' to quit."
)


def main():
    print(BANNER)
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer = answer_question(query)
        print(f"\n{answer}")

    langfuse_client.flush()
    print("Goodbye.")


if __name__ == "__main__":
    main()
