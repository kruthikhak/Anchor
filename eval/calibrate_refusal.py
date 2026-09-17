import json
from pathlib import Path

from rag.retrieval import Retriever

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def main():
    # deliberately a different question set from golden_set.jsonl, otherwise the test score would be tuned on itself
    rows = [json.loads(line) for line in open(EVAL_DIR / "calibration_set.jsonl", encoding="utf-8")]
    retriever = Retriever()

    scored = []
    for row in rows:
        top = retriever.search(row["question"], k=1)[0]
        scored.append((top.rerank_score, row["answerable"], row["question"]))
    scored.sort()

    answerable = [s for s, ok, _ in scored if ok]
    unanswerable = [s for s, ok, _ in scored if not ok]

    # The first version picked the cut-off with the best balanced accuracy (0.50) and refused 6 of 33
    # answerable test questions, while the LLM refused every off-topic question that got past it.
    # A wrongly passed question costs one LLM call, a wrongly refused one loses a correct answer,
    # so the score check now only removes what scores below every answerable calibration question.
    lowest = min(answerable)
    below = [s for s in unanswerable if s < lowest]
    threshold = round((lowest + max(below)) / 2 if below else lowest / 2, 4)
    result = {
        "threshold": threshold,
        "answered": sum(s >= threshold for s in answerable) / len(answerable),
        "refused_before_llm": sum(s < threshold for s in unanswerable) / len(unanswerable),
    }

    for score, ok, question in scored:
        marker = "answerable  " if ok else "not covered "
        side = "to LLM" if score >= threshold else "refuse"
        print(f"{score:.4f}  {marker}  {side}  {question}")

    print(f"\nthreshold {threshold}: sends all answerable questions to the LLM, "
          f"stops {result['refused_before_llm']:.0%} of the uncovered ones before it")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "refusal_threshold.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
