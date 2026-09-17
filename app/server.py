import json
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import config, prompts
from rag.assistant import Assistant, cited_numbers, tidy_citations
from rag.index import STOPWORDS, TOKEN
from rag.llm import chat

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


class PracticeRequest(BaseModel):
    topic: str
    subject: str | None = None


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


@app.post("/api/ask")
def ask(request: AskRequest):
    def events():
        start = time.perf_counter()
        prepared = assistant.prepare(request.question, request.history, docs_for(request.subject))
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
            "retrievalMs": round((time.perf_counter() - start) * 1000),
        })

        answer = ""
        for token in assistant.stream(prepared, request.mode):
            answer += token
            yield sse("token", {"text": token})

        answer = tidy_citations(answer)
        yield sse("done", {
            "answer": answer,
            "citations": cited_numbers(answer, len(prepared.sources)),
            "totalMs": round((time.perf_counter() - start) * 1000),
        })

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/practice")
def practice(request: PracticeRequest):
    prepared = assistant.prepare(request.topic, docs=docs_for(request.subject))
    if not prepared.grounded:
        return {"questions": [], "sources": []}

    reply = chat(
        [
            {"role": "system", "content": prompts.PRACTICE_SYSTEM},
            {"role": "user", "content": prompts.answer_request(request.topic, prepared.sources)},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        reasoning_effort="low",
        max_completion_tokens=900,
    )
    try:
        questions = json.loads(reply.choices[0].message.content or "{}").get("questions", [])
    except json.JSONDecodeError:
        questions = []
    return {
        "questions": questions,
        "sources": [source_json(s.number, s.hit, s.text) for s in prepared.sources],
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
        "models": {"answers": config.ANSWER_MODEL, "judge": "qwen/qwen3.8-27b", "embedder": config.EMBED_MODEL, "reranker": config.RERANK_MODEL},
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
