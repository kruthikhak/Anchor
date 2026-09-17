import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so this runs from any folder

from rag.assistant import Assistant
from rag.grounding import SUPPORT_THRESHOLD, check, summarise

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"


def main():
    answers = json.loads((RESULTS / "generation.json").read_text())
    assistant = Assistant()

    audited, weakest = [], []
    for row in answers:
        if not row["answerable"] or row["refused"]:
            continue
        # retrieval is deterministic, so this rebuilds exactly the passages the answer was written from
        prepared = assistant.prepare(row["question"])
        sources = [{"number": s.number, "text": s.text} for s in prepared.sources]
        sentences = check(row["answer"], sources)
        counts = summarise(sentences)
        audited.append({"id": row["id"], "question": row["question"], **counts})
        weakest += [
            {"id": row["id"], "score": s["bestScore"], "cited": s["cited"], "text": s["text"]}
            for s in sentences
            if not s["ok"]
        ]
        print(f"{row['id']}  {counts['supported']}/{counts['total']} sentences backed by a passage")

    total = sum(a["total"] for a in audited)
    attributed = sum(a["attributed"] for a in audited)
    cited_backed = sum(a["citedBacked"] for a in audited)
    supported = sum(a["supported"] for a in audited)
    weakest.sort(key=lambda s: s["score"] or 0)

    lines = [
        f"Sentence-level grounding over {len(audited)} answers",
        f"a sentence counts as backed when a passage scores at least {SUPPORT_THRESHOLD} against it.",
        "Sentences that cite a source are checked against that source; the rest against every",
        "retrieved passage, because an uncited sentence is often the next step of a credited point.",
        "",
        f"  sentences                  {total}",
        f"  carry a citation           {attributed} ({attributed / total:.0%})",
        f"  cited and backed by it     {cited_backed} ({cited_backed / attributed:.0%} of cited)",
        f"  backed by no passage       {total - supported} ({(total - supported) / total:.0%})",
        "",
        "sentences nothing supports:",
    ]
    for row in weakest[:15]:
        mark = "no citation" if not row["cited"] else f"cites {row['cited']}"
        lines.append(f"  {row['id']}  [{mark}, best {row['score']:.3f}]  {row['text'][:140]}")

    report = "\n".join(lines)
    (RESULTS / "grounding_audit.txt").write_text(report + "\n")
    (RESULTS / "grounding_audit.json").write_text(json.dumps({"answers": audited, "weak": weakest}, indent=2))
    print("\n" + report)


if __name__ == "__main__":
    main()
