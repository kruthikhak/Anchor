import json
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import RateLimitError
from pydantic import BaseModel

from rag import config, prompts
from rag.assistant import Assistant, cited_numbers, is_refusal, tidy_citations
from rag.grounding import check, summarise
from rag.index import STOPWORDS, TOKEN
from rag.llm import chat_json

log = logging.getLogger("anchor")
STATIC = Path(__file__).resolve().parent / "static"
RESULTS = config.ROOT / "eval" / "results"
SOURCES = json.loads(config.CORPUS_FILE.read_text())

assistant = None


@asynccontextmanager
async def lifespan(app):
    global assistant
    assistant = Assistant()
    assistant.retriever.search("warm up")  # load the models now, not during the first question
    yield


app = FastAPI(title="Placement study assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []
    mode: str = "explain"
    subject: str | None = None
    previous: str | None = None  # the reply being built on: for "try another way" and socratic follow-ups


class PracticeRequest(BaseModel):
    topic: str
    subject: str | None = None


class GroundingRequest(BaseModel):
    answer: str
    sources: list[dict]


def docs_for(subject):
    if not subject or subject == "All":
        return None
    return {s["id"] for s in SOURCES if subject in s["subjects"]}


def highlight_terms(query):
    # the words the reader should be able to spot in the passage, roughly what BM25 matched on
    return sorted({w for w in TOKEN.findall(query.lower()) if len(w) > 3 and w not in STOPWORDS})


def source_json(number, hit, text):
    chunk = hit.chunk
    return {
        "number": number,
        "book": chunk.doc_title,
        "docId": chunk.doc_id,
        "section": chunk.section,
        "page": chunk.page_label or str(chunk.page_start),
        "text": text,
        "scores": {
            "rerank": hit.rerank_score,
            "dense": hit.ranks.get("dense"),
            "bm25": hit.ranks.get("bm25"),
        },
    }


def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def source_number(value, count):
    # the model sometimes writes the source as "2" or [2] instead of 2
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= count else None


def answer_events(request):
    start = time.perf_counter()
    prepared = assistant.prepare(request.question, request.history, docs_for(request.subject), request.subject)
    elapsed = lambda: round((time.perf_counter() - start) * 1000)

    if prepared.clarify:
        # each option spells the meaning out, so asking it again can't be ambiguous
        term = prepared.clarify["term"]
        question = request.question.strip().rstrip("?")
        yield sse("clarify", {
            "term": term,
            "options": [{"meaning": m, "question": f"{question} (meaning {m})?"} for m in prepared.clarify["options"]],
        })
        yield sse("done", {"answer": "", "refused": False, "clarify": True, "totalMs": elapsed()})
        return

    if prepared.grounded:
        sources = [source_json(s.number, s.hit, s.text) for s in prepared.sources]
    else:
        # nothing was close enough to answer from, show what came nearest instead
        sources = [source_json(n, hit, hit.chunk.text) for n, hit in enumerate(prepared.closest, start=1)]

    yield sse("sources", {
        "searchQuery": prepared.search_query,
        "grounded": prepared.grounded,
        "terms": highlight_terms(prepared.search_query),
        "sources": sources,
        "corrections": prepared.corrections,
        "expansions": prepared.expansions,
        "retrievalMs": elapsed(),
    })

    answer = prompts.NOT_FOUND
    if prepared.grounded:
        answer = ""
        for token in assistant.stream(prepared, request.mode, request.previous):
            answer += token
            yield sse("token", {"text": token})
    answer = tidy_citations(answer)

    refused = is_refusal(answer)
    if refused:
        # a refusal shouldn't be a dead end: offer the books' own topics that come closest
        yield sse("suggest", assistant.retriever.helper.suggest(prepared.search_query))

    yield sse("done", {
        "answer": answer,
        "refused": refused,
        "citations": cited_numbers(answer, len(prepared.sources)),
        "totalMs": elapsed(),
        "model": prepared.model,
        "fellBack": bool(prepared.model) and prepared.model != config.ANSWER_MODEL,
    })


@app.post("/api/ask")
def ask(request: AskRequest):
    def events():
        try:
            yield from answer_events(request)
        except RateLimitError:
            yield sse("error", {"message": "all three models are out of free-tier tokens for now, which can take a while to reset"})
        except Exception as error:
            # the response has already started streaming, so a failure can only be sent as an event
            log.exception("answer failed")
            yield sse("error", {"message": f"the answer failed ({type(error).__name__})"})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/practice")
def practice(request: PracticeRequest):
    prepared = assistant.prepare(request.topic, docs=docs_for(request.subject), subject=request.subject)
    if not prepared.grounded:
        return {"questions": [], "sources": []}

    reply = chat_json(
        [
            {"role": "system", "content": prompts.PRACTICE_SYSTEM},
            {"role": "user", "content": prompts.answer_request(request.topic, prepared.sources)},
        ],
        temperature=0.4,
        reasoning_effort="low",
        max_completion_tokens=2000,
    )
    questions = reply.get("questions", []) if isinstance(reply, dict) else []
    if not isinstance(questions, list):
        questions = []

    # a question that can't point at one of the sources wasn't written from them, so it's dropped
    kept = []
    for q in questions:
        number = source_number(q.get("source"), len(prepared.sources)) if isinstance(q, dict) else None
        if number and isinstance(q.get("question"), str):
            kept.append({"question": q["question"], "answer": str(q.get("answer", "")), "source": number})
    return {
        "questions": kept,
        "sources": [source_json(s.number, s.hit, s.text) for s in prepared.sources],
    }


@app.post("/api/grounding")
def grounding(request: GroundingRequest):
    # the same cross-encoder that ranked the passages now scores each sentence against what it cites
    rows = check(request.answer, request.sources)
    return {"sentences": rows, **summarise(rows)}


@app.get("/api/library")
def library():
    counts = Counter(c.doc_id for c in assistant.retriever.chunks)
    books = [
        {
            "title": s["title"],
            "authors": s["authors"],
            "subjects": s["subjects"],
            "license": s["license"],
            "url": s["url"] or s.get("note", ""),
            "chunks": counts[s["id"]],
        }
        for s in SOURCES
        if counts[s["id"]]
    ]
    return {"books": books, "chunks": sum(counts.values())}


@app.get("/api/evaluation")
def evaluation():
    retrieval = json.loads((RESULTS / "retrieval.json").read_text())
    answers = json.loads((RESULTS / "generation.json").read_text())
    threshold = json.loads((RESULTS / "refusal_threshold.json").read_text())

    answerable = [a for a in answers if a["answerable"]]
    uncovered = [a for a in answers if not a["answerable"]]
    replied = [a for a in answerable if not a["refused"]]
    share = lambda rows, test: round(sum(1 for r in rows if test(r)) / len(rows), 3) if rows else 0
    # the model the numbers were measured with, which isn't always the one running right now
    measured = Counter(r["model"] for r in answers if r.get("model")).most_common(1)

    return {
        "retrieval": retrieval["summary"],
        "answers": {
            "answerable": len(answerable),
            "uncovered": len(uncovered),
            "answered": share(answerable, lambda r: not r["refused"]),
            "fullyCorrect": share(answerable, lambda r: r.get("correctness") == 2),
            "faithful": share(replied, lambda r: r.get("faithfulness") == 2),
            "cited": share(replied, lambda r: r["citations"]),
            "refusedCorrectly": share(uncovered, lambda r: r["refused"]),
            "stoppedBeforeLLM": sum(1 for r in uncovered if r["refused_before_llm"]),
            "notes": [
                {"id": r["id"], "question": r["question"], "reason": r.get("reason", "")}
                for r in answers
                if r.get("reason") and (r.get("correctness", 2) < 2 or r.get("faithfulness", 2) < 2)
            ],
        },
        "refusalThreshold": threshold["threshold"],
        "models": {"answers": measured[0][0] if measured else config.ANSWER_MODEL, "judge": "qwen/qwen3.8-27b", "embedder": config.EMBED_MODEL, "reranker": config.RERANK_MODEL},
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    # checked with the server on every visit, so a new version's ?v= links always reach the browser
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
