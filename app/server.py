import json
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from itertools import zip_longest
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError, RateLimitError
from pydantic import BaseModel

from rag import config, prompts
from rag.assistant import Assistant, cited_numbers, is_refusal, tidy_citations
from rag.grounding import check, summarise
from rag.index import STOPWORDS, TOKEN
from rag.practice import practice_questions, quiz_marks, quiz_questions
from rag.topics import page_range, table_of_contents, topic_passages

log = logging.getLogger("anchor")
STATIC = Path(__file__).resolve().parent / "static"
RESULTS = config.ROOT / "eval" / "results"
SOURCES = json.loads(config.CORPUS_FILE.read_text())

assistant = None
contents = []


@asynccontextmanager
async def lifespan(app):
    global assistant, contents
    assistant = Assistant()
    assistant.retriever.search("warm up")  # load the models now, not during the first question
    contents = table_of_contents(assistant.retriever.chunks, SOURCES)
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
    subject: str | None = None  # used by the quiz too


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


def passage_json(number, chunk, hit=None, text=None):
    return {
        "number": number,
        "book": chunk.doc_title,
        "docId": chunk.doc_id,
        "section": chunk.section,
        "page": chunk.page_label or str(chunk.page_start),
        "text": text or chunk.text,
        "scores": {
            "rerank": hit.rerank_score if hit else None,
            "dense": hit.ranks.get("dense") if hit else None,
            "bm25": hit.ranks.get("bm25") if hit else None,
        },
    }


def source_json(number, hit, text):
    return passage_json(number, hit.chunk, hit, text)


def clarify_json(question, clarify):
    # each option spells the meaning out, so asking it again can't be ambiguous
    question = question.strip().rstrip("?")
    return {
        "term": clarify["term"],
        "options": [{"meaning": m, "question": f"{question} (meaning {m})?"} for m in clarify["options"]],
    }


def alternatives_json(question, alternatives):
    question = question.strip().rstrip("?")
    return [{"term": term, "meaning": meaning, "subject": subject, "question": f"{question} (meaning {meaning})?"}
            for term, meaning, subject in alternatives]


def model_trouble(error):
    # what a student sees when the practice or quiz model can't deliver
    if isinstance(error, RateLimitError):
        return HTTPException(503, "all three models are out of free-tier tokens for now, which can take a while to reset")
    return HTTPException(503, "the free models couldn't finish writing it just now, so try again in a few minutes")


def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def answer_events(request):
    if not request.question.strip():
        yield sse("error", {"message": "there was no question to answer"})
        return
    start = time.perf_counter()
    prepared = assistant.prepare(request.question, request.history, docs_for(request.subject), request.subject)
    elapsed = lambda: round((time.perf_counter() - start) * 1000)

    if prepared.clarify:
        yield sse("clarify", clarify_json(request.question, prepared.clarify))
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
        "alternatives": alternatives_json(request.question, prepared.alternatives),
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
        yield sse("suggest", assistant.retriever.helper.suggest(prepared.search_query, typed=request.question))

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


def require_topic(topic):
    if not topic.strip():
        raise HTTPException(422, "there was no topic to write questions about")


@app.post("/api/practice")
def practice(request: PracticeRequest):
    require_topic(request.topic)
    prepared = assistant.prepare(request.topic, docs=docs_for(request.subject), subject=request.subject)
    if not prepared.grounded:
        return {"questions": [], "sources": []}
    try:
        questions = practice_questions(request.topic, prepared.sources)
    except (APIError, ValueError) as error:
        raise model_trouble(error)
    return {"questions": questions, "sources": [source_json(s.number, s.hit, s.text) for s in prepared.sources]}


@app.post("/api/quiz")
def quiz(request: PracticeRequest):
    require_topic(request.topic)
    prepared = assistant.prepare(request.topic, docs=docs_for(request.subject), subject=request.subject)
    if prepared.clarify:
        return {"clarify": clarify_json(request.topic, prepared.clarify), "questions": [], "sources": []}
    if not prepared.grounded:
        return {"questions": [], "sources": []}
    marks = quiz_marks(prepared.sources, assistant.retriever.chunks)
    try:
        questions = quiz_questions(request.topic, prepared.sources, marks)
    except (APIError, ValueError) as error:
        raise model_trouble(error)
    return {"marks": marks, "questions": questions, "sources": [source_json(s.number, s.hit, s.text) for s in prepared.sources]}


@app.post("/api/grounding")
def grounding(request: GroundingRequest):
    # the same cross-encoder that ranked the passages now scores each sentence against what it cites
    rows = check(request.answer, request.sources)
    return {"sentences": rows, **summarise(rows)}


@app.get("/api/topics")
def topics():
    return {"books": contents}


@app.get("/api/topic")
def topic(book: str, path: str, start: int = 0):
    found, shown = topic_passages(assistant.retriever.chunks, book, path, max(start, 0))
    if not found:
        raise HTTPException(404, "no such topic")
    return {
        "bookId": book,
        "book": found[0].doc_title,
        "path": path,
        "pages": page_range(found),
        "total": len(found),
        "start": max(start, 0),
        "passages": [{**passage_json(n, chunk), "skip": skip} for n, (chunk, skip) in enumerate(shown, start=max(start, 0) + 1)],
    }


@app.get("/api/search")
def search(q: str, subject: str | None = None):
    """Passages for someone who'd rather read the books than ask. No LLM writes anything here."""
    if not q.strip():
        return {"query": "", "corrections": [], "terms": [], "passages": []}
    understood = assistant.retriever.helper.understand(q, subject)
    # a search box can't ask which meaning of an acronym was meant, so both are searched and interleaved
    if understood.clarify:
        queries = [f"{understood.query} ({meaning})" for meaning in understood.clarify["options"]]
    else:
        queries = [understood.query]
    rankings = [assistant.retriever.search(query, k=8, docs=docs_for(subject)) for query in queries]
    relevant, seen = [], set()
    for hits in zip_longest(*rankings):
        for hit in hits:
            if hit and hit.chunk.id not in seen and hit.rerank_score >= config.REFUSAL_THRESHOLD:
                seen.add(hit.chunk.id)
                relevant.append(hit)
    return {
        "query": " / ".join(queries),
        "corrections": understood.corrections,
        "terms": sorted({t for query in queries for t in highlight_terms(query)}),
        "passages": [source_json(n, h, h.chunk.text) for n, h in enumerate(relevant[:8], start=1)],
    }


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
