# Anchor

A study assistant for placement preparation that answers **only** from nine openly licensed
computer science textbooks, shows the page behind every claim, and says "I couldn't find this in
the study material" when the books don't cover the question.

Built for the Lunorsoft AI Developer assignment, Option 1 (RAG).

- **Ask**: grounded answers that stream in, with clickable citations that open the exact passage,
  four study modes including a marked quiz, three other ways in when an answer doesn't land, and
  practice questions you grade yourself. Select text in any passage to highlight it in your own
  colour or add a note.
- **Topics**: every chapter and section of the books, as their own tables of contents list them,
  filtered by subject. Open one to read it straight from the book, then ask about it, practise it or
  take a quiz on it. The search box finds passages without calling an LLM at all.
- **Library**: the nine books, their subjects and licences
- **My learning**: what you've studied, quiz scores, the answers you marked not quite, the topics
  worth another look, and every highlight and note. It lives in your browser and is never sent
  anywhere.
- **How it's measured**: how well the thing actually works, including what it still gets wrong

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
question ──▶ fix typos, spell out ──▶ search both, fuse the rankings (RRF)
             acronyms                  │
                       rerank candidates (cross-encoder)
                                       │
                    ┌──────────────────┴───────────────────┐
              score too low                          top 5 passages
                    │                                      │
            refuse, offer the                    answer with citations
            closest topics                       (gpt-oss-120b via Groq)
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

**Understanding the question** ([rag/query.py](rag/query.py)) happens before the search.

- *Typos.* A word the books never use and that isn't ordinary English either goes to a small model,
  which decides whether it is a slip ("dedlock") or a real term the books just don't cover
  ("sharding"). A correction only counts if the books use that spelling, so "sharding" is never
  quietly turned into "sharing", and the answer says what it searched for instead.
- *Acronyms.* About thirty placement acronyms are spelled out before searching, because the
  reranker is poor at them: "What does ACID stand for?" scores 0.09 as typed and 0.97 once
  "atomicity, consistency, isolation, durability" is added. When an acronym means two things in these
  books, DSA being data structures and algorithms or the Digital Signature Algorithm in the networks
  book, the chosen subject settles it, and without one the app asks.
- *Refusals that aren't dead ends.* When a question is refused, a model picks up to three of the
  books' own section titles to offer instead: a "did you mean" for a misheard term (mutation →
  Mutexes and Monitors) or the closest topic for an idea the books skip (the banker's algorithm →
  Synchronization and Deadlocks). For a question about something else entirely, like who the
  president is, it offers nothing. Real words that are really slips were the hard case, so the
  code works out which title words look like what was typed ("mutation" looks like "mutex") and
  the model only judges whether that pair is a likely slip.

**Answering** ([rag/assistant.py](rag/assistant.py), [rag/prompts.py](rag/prompts.py)) passes the
top passages to gpt-oss-120b with instructions to use nothing else and cite as it goes. Four study
modes change how the reply is written, and none of them relax the grounding rules:

- *Explain* and *Simple* teach the topic, the second in plainer words and at most about 120 of them.
- *Quiz me* sets a marked quiz from the retrieved passages: multiple choice, fill in the blank and a
  short answer, five marks for a short topic and ten when the book's section runs long. Choices and
  blanks are marked on the page (a typo in a blank still counts), a written answer is compared with
  the model answer, and the score ends with the sections to review. A question that doesn't hold
  up, say a multiple choice whose answer isn't among its options, is dropped before it's shown.
- *Socratic* never hands the answer over. It gives one cited hint and a question, reads the reply,
  says what was right and what is missing, and asks the next question.

"I'm still confused" offers three other ways in, an analogy, a worked example or numbered steps, and
each one sees the reply that didn't land so it doesn't repeat it. Quick practice writes three to
five questions from the same passages, and a question that can't point at one of them is dropped.
You reveal the answer and grade yourself; "not quite" shows why, with a way into a full explanation,
and topics where you missed at least half of your last five answers show up under My learning.

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
| hybrid + section headers | 0.59 | 0.87 | 0.92 | 0.731 | 13 |
| hybrid + reranker | 0.82 | 0.95 | 0.95 | 0.876 | 725 |
| hybrid + headers + reranker | 0.74 | 0.92 | 0.92 | 0.833 | 832 |
| hybrid + headers + small reranker (MiniLM-L6) | 0.64 | 0.92 | 0.92 | 0.774 | 218 |
| **hybrid + reranker, acronyms spelled out** (what the app runs) | **0.85** | **0.95** | **0.95** | **0.897** | 806 |

The reranker is the single biggest win and the single biggest cost: it takes retrieval from 0.56 to
0.82 at rank one, and it is roughly fifty times slower than the searches feeding it. A candidate
pool of 20 scores the same as 30 and reranks about 40% faster on CPU, so the app uses 20.

Spelling acronyms out first moved three questions' evidence up (TCP vs UDP from third to first) and
one down (BFS from first to second), a net gain at rank one.

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
| Fully correct, judged against the reference answer | 87% (34/39) |
| Every claim backed by the passages, judged | 92% (36/39) |
| Answers citing at least one source | 100% |
| Questions outside the library correctly refused | 10/10 |

These are measured the way the app runs, acronyms spelled out, which is what took the ACID answer
from partly to fully correct and cleared the unsupported claims out of TCP vs UDP and DDL vs DML
(85% and 87% before). The judge is `qwen/qwen3.8-27b`, a different model family from the one writing
the answers, so it isn't marking its own style. Every LLM call in the evaluation is cached on disk by
its exact request, so rerunning `python eval/run_generation.py` reproduces the numbers above exactly,
in under a minute and without spending any API budget.

### Every sentence, checked against its source

`python eval/audit_grounding.py` scores each sentence of an answer against the passage it cites,
reusing the cross-encoder that ranked the passages in the first place. Sentences without a citation
are scored against every retrieved passage instead, because an uncited sentence is usually the next
step of a point already credited rather than an invention. No LLM is involved, so this measure is
deterministic and independent of the judge above.

| Across 267 sentences from 39 answers | Result |
|---|---|
| Sentences carrying a citation | 51% |
| Cited sentences backed by what they cite | 94% |
| Sentences backed by no retrieved passage | 9% |

That last 9% is the part worth reading. It holds the genuine problems, such as a B-tree answer
claiming that multi-dimensional indexing "would be inefficient with a BST", which no passage says,
and a sentence whose citation points at a passage that doesn't back it. It also has false positives:
terse steps like "Push x to the back of the deque", and even a stack-of-plates analogy that Open
Data Structures really does use. So the number is a reading list, not a verdict. The app shows the
same count under every answer and can highlight the weak sentences in place.

### Two things the evaluation caught

- **The refusal threshold was too strict.** Chosen for best balanced accuracy on a separate
  calibration set, it refused 6 of 39 answerable test questions, while the LLM refused every
  off-topic question that reached it. Since the model is a reliable second layer, a wrongly passed
  question costs one call and a wrongly refused one loses a correct answer, so the threshold now
  only blocks what scores below every answerable calibration question. Wrong refusals went to zero
  and out-of-scope refusals stayed at 10/10.
- **Restructuring the prompt cost faithfulness.** Adding study modes meant reorganising the system
  prompt. Two reorganised versions scored 74% and 72% on faithfulness against 87% for the original.
  Each ran once with answers sampled at temperature 0.3, so part of that gap may be run-to-run noise,
  but both landed well below. My first explanation (the order of the instructions) was wrong: moving
  the grounding rules to the end made it no better. Explain mode, the default and the one measured
  here, now uses the original prompt word for word, and the other modes swap only its final style
  paragraph.

## What it still gets wrong

- A few answers add small details the passages don't state, such as the primary and foreign key
  answer describing how a diagram draws the link. The How it's measured page lists every one the
  judge flagged, and the grounding audit above finds the same cases without asking an LLM.
- The OSI answer names the first four layers and then says the books don't name the other three,
  though one of its passages does.
- The write-ahead logging answer gets the order of commit and data writes wrong in every run so far.
- Two questions retrieve the wrong passages: why file systems prefer B-trees, and what isolation
  means for transactions.
- The reranker is poorly calibrated for acronyms. Spelling out the ones in the table fixes those,
  but an acronym missing from it still scores low, which is why the refusal threshold has to sit low.
- Typo fixing leaves real words alone, so "trashing" for thrashing isn't corrected.
- The topics offered after a refusal are a model's pick and can miss. The smaller fallback model is
  less sure of itself with a lone word like "mutation", so a single word that looks like a title
  word always gets those titles offered, labelled as similar-looking rather than as what was meant.
- Quiz and practice questions are only as good as the model writing them. They're checked for
  shape (a source, a gap in a blank, an answer among the options), not for truth, and the smaller
  model once wrote a blank whose answer contradicted its own passage.
- Progress, highlights and notes are kept in the browser, so they don't follow you to another device.
- Groq's free tier allows 200k tokens a day per model. When gpt-oss-120b runs out, answers come from
  gpt-oss-20b and then qwen3.8-27b, and each answer says which model wrote it. The evaluation
  measured gpt-oss-120b only.

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

While developing, `ANSWER_MODEL=openai/gpt-oss-20b uvicorn app.server:app` saves the larger
model's daily allowance.

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

Check the logic that needs neither a model nor the API (spelling and acronym handling, citations,
refusals, the model fallback order):

```bash
python -m unittest discover tests
```

## Layout

```
rag/          ingestion, chunking, indexing, retrieval, question understanding, prompts,
              answering, grounding check, the books' tables of contents, practice and quizzes
app/          FastAPI server and a small front end in plain JavaScript, no framework or build step
scripts/      download the corpus, build the index, ask from the terminal
eval/         test questions, the four measurement scripts, and their results
tests/        unit tests, none of which call the API
data/         corpus.json lists every book; PDFs and the index are built, not committed
deploy/       Dockerfile and notes for a Hugging Face Space
```

## AI tools used

The brief allows AI tools provided their use is disclosed, so: this project was built with Claude
Code (Anthropic) as a pair programmer. I chose the problem, the corpus and the evaluation design,
decided every tradeoff recorded above, hand-checked the test questions against the books, and ran
and reviewed everything in this repository. Claude Code wrote much of the implementation to that
direction, and found two of the bugs listed above while I was testing. The web front end was built
with its help, from a design direction and feature list I set.

The application itself uses `openai/gpt-oss-120b` through Groq to write answers and pick topic
suggestions, `openai/gpt-oss-20b` to rewrite follow-up questions and check spelling,
`BAAI/bge-base-en-v1.5` for embeddings and `BAAI/bge-reranker-base` for reranking. When a model's
free-tier allowance runs out it falls back to the other gpt-oss model and then `qwen/qwen3.8-27b`.
The evaluation judge is `qwen/qwen3.8-27b`; the evaluation calls its models directly and never
falls back.

## Licence

The code is MIT. The textbooks keep their own licences, listed above and in `data/corpus.json`;
none of them are redistributed in this repository.
