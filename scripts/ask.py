import argparse
import time

from rag.assistant import Assistant, cited_numbers


def main():
    parser = argparse.ArgumentParser(description="Ask the study assistant a question from the terminal")
    parser.add_argument("question")
    parser.add_argument("--method", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    assistant = Assistant(method=args.method, rerank=not args.no_rerank)

    start = time.time()
    prepared = assistant.prepare(args.question)
    print(f"retrieval took {time.time() - start:.2f}s\n")
    for s in prepared.sources:
        c = s.hit.chunk
        score = f"{s.hit.rerank_score:.3f}" if s.hit.rerank_score is not None else "-"
        print(f"[{s.number}] {c.doc_title} | {c.section} | p.{c.page_label or c.page_start} | rerank {score} | ranks {s.hit.ranks}")

    if args.retrieve_only:
        return

    print()
    start = time.time()
    answer = ""
    for token in assistant.stream(prepared):
        answer += token
        print(token, end="", flush=True)
    print(f"\n\ngeneration took {time.time() - start:.1f}s, cited {cited_numbers(answer, len(prepared.sources))}")


if __name__ == "__main__":
    main()
