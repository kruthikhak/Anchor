import re
from dataclasses import dataclass, field

from . import config, prompts
from .llm import chat
from .retrieval import Hit, Retriever

CITATION = re.compile(r"\[(\d+)\]")
CODE_BLOCK = re.compile(r"```.*?```", re.S)
# gpt-oss was trained to cite search results like 【1†L1-L4】 and sometimes does despite the prompt
OSS_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")

GENERATION = {"reasoning_effort": "low", "temperature": 0.3, "max_completion_tokens": 1500}


@dataclass
class Source:
    number: int
    hit: Hit
    text: str
    page_end: int


@dataclass
class Prepared:
    question: str
    search_query: str
    sources: list = field(default_factory=list)
    grounded: bool = True  # False when nothing retrieved was relevant enough to answer from
    closest: list = field(default_factory=list)


def join_overlapping(first, second):
    # consecutive chunks share a few sentences of overlap, don't show them twice
    for size in range(min(len(first), len(second), 800), 20, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return first + "\n" + second


def tidy_citations(answer):
    return OSS_CITATION.sub(r"[\1]", answer)


def cited_numbers(answer, n_sources):
    prose = CODE_BLOCK.sub("", tidy_citations(answer))  # a[1] inside code is not a citation
    return sorted({int(n) for n in CITATION.findall(prose) if 1 <= int(n) <= n_sources})


class Assistant:
    def __init__(self, retriever=None, method="hybrid", rerank=True, k=5, expand=0, min_score=config.REFUSAL_THRESHOLD):
        self.retriever = retriever or Retriever()
        self.method = method
        self.rerank = rerank
        self.k = k
        # how many of the top hits get their following chunk attached; off by default because it
        # didn't change how often the evidence reached the LLM (eval/results/retrieval_summary.txt)
        self.expand = expand
        self.min_score = min_score

    def rewrite(self, question, history):
        if not history:
            return question
        response = chat(
            [
                {"role": "system", "content": prompts.REWRITE_SYSTEM},
                {"role": "user", "content": prompts.rewrite_request(question, history)},
            ],
            model=config.REWRITE_MODEL,
            reasoning_effort="low",
            max_completion_tokens=400,
        )
        return (response.choices[0].message.content or "").strip() or question

    def prepare(self, question, history=None):
        query = self.rewrite(question, history)
        hits = self.retriever.search(query, method=self.method, rerank=self.rerank, k=self.k)

        # the score check needs reranker scores, so it's skipped when reranking is switched off
        if not hits or (self.rerank and hits[0].rerank_score < self.min_score):
            return Prepared(question, query, grounded=False, closest=hits[:3])
        return Prepared(question, query, self.build_sources(hits))

    def build_sources(self, hits):
        sources = []
        used = {h.chunk.id for h in hits}
        for number, hit in enumerate(hits, start=1):
            text, page_end = hit.chunk.text, hit.chunk.page_end
            if number <= self.expand:
                # answers often run past a chunk boundary, e.g. a list of four conditions split two and two
                following = self.retriever.next_chunk(hit.chunk)
                if following and following.section == hit.chunk.section and following.id not in used:
                    text = join_overlapping(text, following.text)
                    page_end = following.page_end
                    used.add(following.id)
            sources.append(Source(number, hit, text, page_end))
        return sources

    @staticmethod
    def messages(prepared):
        return [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": prompts.answer_request(prepared.search_query, prepared.sources)},
        ]

    def stream(self, prepared):
        if not prepared.grounded:
            yield prompts.NOT_FOUND
            return
        for event in chat(self.messages(prepared), stream=True, **GENERATION):
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    def answer(self, question, history=None):
        prepared = self.prepare(question, history)
        return prepared, tidy_citations("".join(self.stream(prepared)))
