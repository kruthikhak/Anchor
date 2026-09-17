import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from openai import RateLimitError

from rag import config, prompts
from rag.assistant import GENERATION, Assistant, cited_numbers, tidy_citations
from rag.llm import client

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# a different model family from the one answering, so the judge isn't marking its own style
JUDGE_MODEL = "qwen/qwen3.8-27b"

JUDGE_PROMPT = """You are grading a study assistant that must answer only from the sources it was given.

Question:
{question}

Reference answer, written by a person from the same textbooks:
{reference}

{sources}

Assistant's answer:
{answer}

Grade two things.

correctness, comparing the assistant's answer with the reference answer:
2 = covers the key points of the reference with no errors
1 = partly right, or misses an important point
0 = wrong

faithfulness, checking every factual claim in the assistant's answer against the sources (not the reference):
2 = every claim is supported by the sources
1 = mostly supported, one minor detail is not
0 = makes a significant claim the sources do not support

Reply with JSON only: {{"correctness": 0, "faithfulness": 0, "reason": "one short sentence"}}"""


class CachedLLM:
    # eval runs get repeated while tuning, and the free tier is slow, so responses are kept on disk
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists calls (key text primary key, response text)")

    def __call__(self, **request):
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        row = self.db.execute("select response from calls where key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])

        for attempt in range(6):
            try:
                response = client().chat.completions.create(**request)
                break
            except RateLimitError:
                time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"still rate limited on {request['model']}")

        result = {"content": response.choices[0].message.content or "", "model": response.model}
        self.db.execute("insert into calls values (?, ?)", (key, json.dumps(result)))
        self.db.commit()
        return result


def normalise(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_refusal(answer):
    return normalise(answer).startswith(normalise(prompts.NOT_FOUND))


def main():
    rows = [json.loads(line) for line in open(EVAL_DIR / "golden_set.jsonl", encoding="utf-8")]
    threshold = json.loads((RESULTS_DIR / "refusal_threshold.json").read_text())["threshold"]
    assistant = Assistant(min_score=threshold)
    llm = CachedLLM(EVAL_DIR / "cache" / "llm.sqlite")

    results = []
    for row in rows:
        prepared = assistant.prepare(row["question"])
        if prepared.grounded:
            reply = llm(model=config.ANSWER_MODEL, messages=assistant.messages(prepared), **GENERATION)
            answer, model = tidy_citations(reply["content"]), reply["model"]
        else:
            answer, model = prompts.NOT_FOUND, None

        result = {
            "id": row["id"],
            "subject": row["subject"],
            "question": row["question"],
            "answerable": row["answer"] is not None,
            "answer": answer,
            "model": model,
            "refused_before_llm": not prepared.grounded,
            "refused": is_refusal(answer),
            "citations": cited_numbers(answer, len(prepared.sources)),
        }

        if result["answerable"] and not result["refused"]:
            verdict = llm(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=row["question"],
                    reference=row["answer"],
                    sources=prompts.answer_request(prepared.search_query, prepared.sources).rsplit("\n\nQuestion:", 1)[0],
                    answer=answer,
                )}],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=300,
            )
            result.update(json.loads(verdict["content"]))

        results.append(result)
        print(f"{row['id']}  refused={result['refused']}  correctness={result.get('correctness', '-')}  faithfulness={result.get('faithfulness', '-')}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "generation.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    report = summary(results, threshold)
    (RESULTS_DIR / "generation_summary.txt").write_text(report + "\n")
    print("\n" + report)


def summary(results, threshold):
    answerable = [r for r in results if r["answerable"]]
    uncovered = [r for r in results if not r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]
    pct = lambda n, d: f"{n}/{d} ({n / d:.0%})" if d else "0/0"

    lines = [
        f"Answers on {len(answerable)} answerable and {len(uncovered)} not-covered questions",
        f"answer model {config.ANSWER_MODEL}, judge {JUDGE_MODEL}, refusal threshold {threshold}",
        "",
        "answerable questions",
        f"  answered instead of refusing   {pct(len(answered), len(answerable))}",
        f"  fully correct (2/2)            {pct(sum(r.get('correctness') == 2 for r in answerable), len(answerable))}",
        f"  at least partly correct        {pct(sum((r.get('correctness') or 0) >= 1 for r in answerable), len(answerable))}",
        f"  fully faithful to sources      {pct(sum(r.get('faithfulness') == 2 for r in answered), len(answered))}",
        f"  cite at least one source       {pct(sum(bool(r['citations']) for r in answered), len(answered))}",
        "",
        "questions the books don't cover",
        f"  correctly refused              {pct(sum(r['refused'] for r in uncovered), len(uncovered))}",
        f"    stopped by retrieval score   {sum(r['refused_before_llm'] for r in uncovered)}",
        f"    refused by the LLM           {sum(r['refused'] and not r['refused_before_llm'] for r in uncovered)}",
    ]

    problems = [r for r in answerable if r["refused"] or r.get("correctness", 2) < 2 or r.get("faithfulness", 2) < 2]
    problems += [r for r in uncovered if not r["refused"]]
    if problems:
        lines += ["", "worth a look:"]
        for r in problems:
            note = r.get("reason") or ("refused" if r["refused"] else "answered a question the books don't cover")
            lines.append(f"  {r['id']}  {r['question']}  ->  {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
