# Anchor

A study assistant for placement preparation that answers **only** from nine openly licensed
computer science textbooks, shows the page behind every claim, and says "I couldn't find this in
the study material" when the books don't cover the question.

Built for the Lunorsoft AI Developer assignment, Option 1 (RAG).

- Ask page: grounded answers that stream in, with clickable citations that open the exact passage
- Library page: the nine books, their subjects and licences
- Evaluation page: how well the thing actually works, measured, including what it still gets wrong

## Why it is built this way

Most "chat with your PDF" demos work because nobody checks them. The interesting part of a RAG
system is not that it produces an answer, it is whether the answer came from the source it claims
and whether the system knows when to stay quiet. So the whole project is arranged around three
questions, each with a number attached:

1. Does retrieval find the passage that actually answers the question?
2. Is every sentence of the answer backed by the passage it cites?
3. Does it refuse when the library doesn't cover the topic?

## How it works

```
9 PDFs ──▶ extract ──▶ chunk ──▶ ┌ dense index (FAISS)  ┐
          (PyMuPDF)   (~380 tok) └ keyword index (BM25) ┘
                                       │
                              fuse both rankings (RRF)
                                       │
                       rerank candidates (cross-encoder)
                                       │
                    ┌──────────────────┴───────────────────┐
              score too low                          top 5 passages
                    │                                      │
            refuse, show the                     answer with citations
            closest passages                     (gpt-oss-120b via Groq)
                                                           │
                                            check each sentence against
                                            the passage it cites
```

**Ingestion** ([rag/ingest.py](rag/ingest.py)) walks each PDF's own table of contents so every
paragraph keeps its section path, which is what makes a citation like
"Synchronization and Deadlocks › Deadlock › The Deadlock Problem, page 126" possible. It detects
each book's running heads statistically (blocks repeating at the same height on a quarter of the
pages, mostly containing a digit) rather than assuming fixed margins, and drops index pages,
bibliographies and end-of-chapter exercises. Exercises matter: they are question-shaped, so they
rank well for question-shaped queries and never contain answers. Dropping them removed 204 chunks
from one book alone.

**Chunking** ([rag/chunking.py](rag/chunking.py)) packs paragraphs to about 380 tokens, measured
with the embedding model's own tokenizer, prefers to start a chunk at a heading, and carries about
60 tokens of overlap so an explanation split across a boundary isn't lost.

**Retrieval** ([rag/retrieval.py](rag/retrieval.py)) runs BM25 and dense search separately, fuses
them with reciprocal rank fusion, and reranks with a cross-encoder. One detail worth calling out:
each retriever nominates its own top candidates for reranking. Cutting the pool by fused order
instead drops a chunk that only one retriever found, even when it was that retriever's first
result, in favour of chunks both ranked as mediocre.

**Answering** ([rag/assistant.py](rag/assistant.py), [rag/prompts.py](rag/prompts.py)) passes the
top passages to gpt-oss-120b with instructions to use nothing else and cite as it goes. Four study
modes change how the reply is written (explain, simple, quiz, socratic); none of them relax the
grounding rules.

**Refusing** is two layers. A question whose best passage reranks below a calibrated threshold never
reaches the LLM. Anything above it goes to the model, which refuses on its own when the passages
don't answer the question. Both layers earn their place in the numbers below.

## What the numbers say

Measured on 39 placement-style questions written the way an interviewer asks them, each with
verbatim quotes from the books as the answer key, plus 10 questions the books don't cover.
Reproduce with `python eval/run_retrieval.py` and `python eval/run_generation.py`.

### Retrieval: what each step is worth

| Setup | top 1 | top 3 | top 5 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| dense only | 0.56 | 0.87 | 0.87 | 0.717 | 18 |
| BM25 only | 0.64 | 0.90 | 0.90 | 0.756 | 1 |
| hybrid (BM25 + dense, RRF) | 0.69 | 0.90 | 0.90 | 0.793 | 13 |
| hybrid + section headers | 0.59 | 0.87 | 0.92 | 0.731 | 14 |
| **hybrid + reranker** (what the app runs) | **0.82** | **0.95** | **0.95** | **0.876** | 844 |
| hybrid + headers + reranker | 0.74 | 0.92 | 0.92 | 0.833 | 1139 |
| hybrid + headers + small reranker (MiniLM-L6) | 0.64 | 0.92 | 0.92 | 0.774 | 313 |

The reranker is the single biggest win and the single biggest cost: it takes retrieval from 0.56 to
0.82 at rank one, and it is roughly fifty times slower than the searches feeding it. A candidate
pool of 20 scores the same as 30 and reranks about 40% faster on CPU, so the app uses 20.

### Two ideas that did not work

Both are kept in the table above rather than quietly deleted.

- **Section headers in the indexed text.** Prefixing each chunk with its book and section made
  fusion clearly worse (0.69 → 0.59 at rank one). The likely reason is that the header repeats the
  same words on every chunk of a section, which blunts the keyword search that was doing the most
  work.
- **Attaching the following chunk to each top hit.** Intended to catch answers that run past a
  chunk boundary; it left the share of questions whose evidence reached the model unchanged at 0.95,
  while making every request longer, so it is off by default.

### Answers

| | Result |
|---|---|
| Answerable questions answered rather than wrongly refused | 39/39 |
| Fully correct, judged against the reference answer | 85% |
| Every claim backed by the passages, judged | 87% |
| Answers citing at least one source | 100% |
| Questions outside the library correctly refused | 10/10 |

The judge is `qwen/qwen3.8-27b`, a different model family from the one writing the answers, so it
isn't marking its own style. Every LLM call in the evaluation is cached on disk, so a rerun costs
nothing unless something actually changed.

### Two things the evaluation caught

- **The refusal threshold was too strict.** Chosen for best balanced accuracy on a separate
  calibration set, it refused 6 of 39 answerable test questions, while the LLM refused every
  off-topic question that reached it. Since the model is a reliable second layer, a wrongly passed
  question costs one call and a wrongly refused one loses a correct answer, so the threshold now
  only blocks what scores below every answerable calibration question. Wrong refusals went to zero
  and out-of-scope refusals stayed at 10/10.
- **Prompt order changes faithfulness.** Splitting the system prompt so the writing style came last
  dropped faithfulness from 87% to 74%. Moving the grounding rules back to the end recovered it.

## What it still gets wrong

- A few answers add small details the passages don't state, such as calling UDP "simplex per
  datagram". The evaluation page lists every one the judge flagged.
- The write-ahead logging answer gets the order of commit and data writes wrong in every run so far.
- Two questions retrieve the wrong passages: why file systems prefer B-trees, and what isolation
  means for transactions.
- The reranker is poorly calibrated for acronym questions. "What does ACID stand for?" scores 0.09
  even though a passage spells the acronym out, which is why the refusal threshold has to sit low.

## The library

| Book | Authors | Subjects | Licence |
|---|---|---|---|
| Algorithms | Jeff Erickson | DSA | CC BY 4.0 |
| Hash Tables, Disjoint Sets, Amortized Analysis (lecture notes) | Jeff Erickson | DSA | CC BY 4.0 |
| Open Data Structures | Pat Morin | DSA, DBMS | CC BY 2.5 |
| Competitive Programmer's Handbook | Antti Laaksonen | DSA | CC BY-NC-SA 4.0 |
| Operating Systems and Middleware | Max Hailperin | OS, DBMS | CC BY-SA 3.0 |
| Computer Networks: A Systems Approach | Peterson, Davie | Networks | CC BY 4.0 |
| Database Design, 2nd edition | Watt, Eng | DBMS | CC BY 4.0 |

Object-oriented programming, system design and aptitude are deliberately out of scope, so questions
about them are refused rather than answered from general knowledge.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_corpus.py     # fetches 8 of the 9 books, the 9th is noted in data/corpus.json
python scripts/build_index.py         # chunks and embeds, about 10 minutes
cp .env.example .env                  # add a Groq API key
uvicorn app.server:app --port 8000
```

Ask from the terminal instead of the browser:

```bash
python scripts/ask.py "What conditions must hold for a deadlock to occur?"
python scripts/ask.py                 # interactive, with follow-up questions
```

Re-run the measurements:

```bash
python eval/run_retrieval.py          # the ablation table
python eval/calibrate_refusal.py      # the refusal threshold
python eval/run_generation.py         # answer quality, judged
python eval/audit_grounding.py        # sentence-level grounding
```

## Layout

```
rag/          ingestion, chunking, indexing, retrieval, prompts, answering, grounding check
app/          FastAPI server and a small front end with no framework and no CDN
scripts/      download the corpus, build the index, ask from the terminal
eval/         test questions, the four measurement scripts, and their results
data/         corpus.json lists every book; PDFs and the index are built, not committed
deploy/       Dockerfile and notes for a Hugging Face Space
```

## AI tools used

The brief allows AI tools provided their use is disclosed, so: this project was built with Claude
Code (Anthropic) as a pair programmer. I chose the problem, the corpus and the evaluation design,
decided every tradeoff recorded above, hand-checked the test questions against the books, and ran
and reviewed everything in this repository. Claude Code wrote much of the implementation to that
direction, and found two of the bugs listed above while I was testing.

The application itself uses `openai/gpt-oss-120b` through Groq to write answers,
`BAAI/bge-base-en-v1.5` for embeddings and `BAAI/bge-reranker-base` for reranking. The evaluation
judge is `qwen/qwen3.8-27b`.

## Licence

The code is MIT. The textbooks keep their own licences, listed above and in `data/corpus.json`;
none of them are redistributed in this repository.
