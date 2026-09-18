import re

from .index import reranker

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")
CODE_BLOCK = re.compile(r"```.*?```", re.S)
LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
EMPHASIS = re.compile(r"\*{1,2}")
CITATION = re.compile(r"\[(\d{1,2})\]")

# a cited passage the cross-encoder scores below this isn't really backing the sentence;
# picked from the score distribution in eval/results/grounding_audit.txt
SUPPORT_THRESHOLD = 0.05


def sentences(answer):
    prose = CODE_BLOCK.sub(" ", answer)
    out = []
    for line in prose.split("\n"):
        line = EMPHASIS.sub("", LIST_MARK.sub("", line.strip()))
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        if line.lower().startswith("analogy"):
            continue  # an everyday comparison the student asked for, openly not from the books
        if len(line.split()) < 8 and not line.endswith((".", "!", "?")):
            continue  # a bold label or a heading, not a claim to check
        # the questions in a socratic reply ask, they don't claim anything
        out.extend(s.strip() for s in SENTENCE_SPLIT.split(line) if len(s.strip()) > 25 and not s.strip().endswith("?"))
    return out


def check(answer, sources):
    """Score every sentence against the passages: the ones it cites, and the rest.

    A sentence with no citation is not automatically a problem, it is often the next step of a
    point already credited. What matters is whether any retrieved passage backs it at all."""
    passages = {int(s["number"]): s["text"] for s in sources}
    rows, pairs, owners = [], [], []

    for sentence in sentences(answer):
        cited = [int(n) for n in CITATION.findall(sentence) if int(n) in passages]
        rows.append({"text": sentence, "cited": cited, "citedScore": None, "bestScore": None, "bestSource": None})
        # a cited sentence only needs checking against what it cites, which roughly halves the
        # cross-encoder work, and on a two-core host that is the slow part
        for number in cited or passages:
            pairs.append((CITATION.sub("", sentence).strip(), passages[number]))
            owners.append((len(rows) - 1, number))

    if pairs:
        for (row_index, number), score in zip(owners, reranker().predict(pairs, batch_size=32)):
            row = rows[row_index]
            score = float(score)
            if score > (row["bestScore"] if row["bestScore"] is not None else -1):
                row["bestScore"], row["bestSource"] = score, number
            if number in row["cited"]:
                row["citedScore"] = max(row["citedScore"] or 0.0, score)

    for row in rows:
        against = row["citedScore"] if row["cited"] else row["bestScore"]
        row["ok"] = against is not None and against >= SUPPORT_THRESHOLD
    return rows


def summarise(rows):
    cited = [r for r in rows if r["cited"]]
    return {
        "total": len(rows),
        "attributed": len(cited),
        "citedBacked": sum(1 for r in cited if r["ok"]),
        "supported": sum(1 for r in rows if r["ok"]),
    }
