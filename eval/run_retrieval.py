import json
import re
import statistics
import time
from pathlib import Path

from rag import config
from rag.assistant import Assistant
from rag.index import device, embedder, reranker
from rag.retrieval import Retriever

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

SMALL_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Each row changes one thing against an earlier row, so every row shows what that change is worth:
# C combines A and B, D adds section headers to C, E and F add the reranker to C and D,
# G swaps F's reranker for one small enough to be fast on a free CPU host.
CONFIGS = [
    ("A", "dense only", dict(contextual=False, method="dense", rerank=False)),
    ("B", "BM25 only", dict(contextual=False, method="bm25", rerank=False)),
    ("C", "hybrid (BM25 + dense, RRF)", dict(contextual=False, method="hybrid", rerank=False)),
    ("D", "C + section headers", dict(contextual=True, method="hybrid", rerank=False)),
    ("E", "C + reranker (bge-reranker-base)", dict(contextual=False, method="hybrid", rerank=True, rerank_model=config.RERANK_MODEL)),
    ("F", "D + reranker (bge-reranker-base)", dict(contextual=True, method="hybrid", rerank=True, rerank_model=config.RERANK_MODEL)),
    ("G", "F with a small reranker (MiniLM-L6)", dict(contextual=True, method="hybrid", rerank=True, rerank_model=SMALL_RERANKER)),
]


def normalise(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def rank_of_evidence(texts, quotes):
    for rank, text in enumerate(texts, start=1):
        text = normalise(text)
        if any(q in text for q in quotes):
            return rank
    return None


def summarise(ranks, latencies):
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n
    return {
        "hit@1": hit(1),
        "hit@3": hit(3),
        "hit@5": hit(5),
        "mrr@10": sum(1 / r for r in ranks if r is not None and r <= 10) / n,
        "ms_per_query": statistics.median(latencies) * 1000,
    }


def main():
    rows = [json.loads(line) for line in open(EVAL_DIR / "golden_set.jsonl", encoding="utf-8")]
    questions = [r for r in rows if r["evidence"]]
    retrievers = {flag: Retriever(contextual=flag) for flag in (False, True)}

    corpus = [normalise(c.text) for c in retrievers[False].chunks]
    for q in questions:
        q["quotes"] = [normalise(e) for e in q["evidence"]]
        if not any(quote in text for quote in q["quotes"] for text in corpus):
            print(f"warning: evidence for {q['id']} not found anywhere in the corpus")

    # load the models and warm them up so the first query isn't charged for it
    embedder().encode(["warm up"])
    for name in (config.RERANK_MODEL, SMALL_RERANKER):
        reranker(name).predict([("warm up", "warm up")])

    per_question = {q["id"]: {"question": q["question"]} for q in questions}
    summary = {}
    for key, name, cfg in CONFIGS:
        retriever = retrievers[cfg["contextual"]]
        ranks, latencies = [], []
        for q in questions:
            start = time.perf_counter()
            hits = retriever.search(q["question"], method=cfg["method"], rerank=cfg["rerank"], k=10,
                                    rerank_model=cfg.get("rerank_model", config.RERANK_MODEL))
            latencies.append(time.perf_counter() - start)
            rank = rank_of_evidence([h.chunk.text for h in hits], q["quotes"])
            ranks.append(rank)
            per_question[q["id"]][key] = rank
        summary[key] = {"name": name, **summarise(ranks, latencies)}
        print(f"{key} done")

    # context recall: is the evidence anywhere in the text the LLM actually gets to read
    for expand in (0, 2):
        assistant = Assistant(retriever=retrievers[False], k=5, expand=expand)
        found = 0
        for q in questions:
            hits = assistant.retriever.search(q["question"], k=5)
            texts = [s.text for s in assistant.build_sources(hits)]
            found += rank_of_evidence(texts, q["quotes"]) is not None
        summary[f"context_recall_expand_{expand}"] = found / len(questions)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps({"summary": summary, "questions": per_question}, indent=2))

    n_docs = len({c.doc_id for c in retrievers[False].chunks})
    lines = [
        f"Retrieval on {len(questions)} hand-checked questions, {len(corpus)} chunks from {n_docs} documents",
        f"latency measured on this machine ({device()})",
        "",
    ]
    lines.append(f"{'config':<38}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR@10':>8}{'ms/query':>10}")
    for key, name, _ in CONFIGS:
        s = summary[key]
        lines.append(f"{key + '  ' + name:<38}{s['hit@1']:>7.2f}{s['hit@3']:>7.2f}{s['hit@5']:>7.2f}{s['mrr@10']:>8.3f}{s['ms_per_query']:>10.0f}")
    lines.append("")
    lines.append(f"evidence inside the LLM context (config E, top 5): {summary['context_recall_expand_0']:.2f}")
    lines.append(f"same, with the next chunk attached to the top 2:   {summary['context_recall_expand_2']:.2f}")
    lines.append("")
    lines.append("misses (rank of the evidence chunk per config, - means not in the top 10):")
    for qid, row in per_question.items():
        if any(row[k] is None or row[k] > 3 for k, _, _ in CONFIGS):
            ranks = "  ".join(f"{k}={row[k] or '-'}" for k, _, _ in CONFIGS)
            lines.append(f"  {qid}  {ranks}  {row['question']}")

    report = "\n".join(lines)
    (RESULTS_DIR / "retrieval_summary.txt").write_text(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
