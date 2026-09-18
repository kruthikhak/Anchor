import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so this runs from any folder

from rag.assistant import Assistant, cited_numbers, tidy_citations


def show_sources(prepared):
    if prepared.grounded:
        hits = [s.hit for s in prepared.sources]
    else:
        print("nothing in the library scored above the refusal threshold, closest passages were:")
        hits = prepared.closest
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        score = f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "-"
        print(f"[{n}] {c.doc_title} | {c.section} | p.{c.page_label or c.page_start} | rerank {score} | ranks {hit.ranks}")


def ask(assistant, question, history, retrieve_only):
    start = time.time()
    prepared = assistant.prepare(question, history)
    if prepared.clarify:
        options = " or ".join(prepared.clarify["options"])
        print(f"{prepared.clarify['term']} could mean {options}. Ask again with the one you mean spelled out.")
        return ""
    if prepared.search_query != question:
        print(f"searching for: {prepared.search_query}")
    print(f"retrieval took {time.time() - start:.2f}s\n")
    show_sources(prepared)
    if retrieve_only:
        return ""

    print()
    start = time.time()
    answer = ""
    for token in assistant.stream(prepared):
        answer += token
        print(token, end="", flush=True)
    answer = tidy_citations(answer)
    print(f"\n\ngeneration took {time.time() - start:.1f}s, cited {cited_numbers(answer, len(prepared.sources))}")
    return answer


def main():
    parser = argparse.ArgumentParser(description="Ask the study assistant from the terminal")
    parser.add_argument("question", nargs="?", help="leave it out to ask several questions in a row, with follow-ups")
    parser.add_argument("--method", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true", help="show the retrieved passages without calling the LLM")
    args = parser.parse_args()

    assistant = Assistant(method=args.method, rerank=not args.no_rerank)
    if args.question:
        ask(assistant, args.question, None, args.retrieve_only)
        return

    print("Ask a question, or press Enter on an empty line to quit. Follow-ups use the conversation so far.")
    history = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer = ask(assistant, question, history, args.retrieve_only)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
