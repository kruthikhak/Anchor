import re
from dataclasses import dataclass
from itertools import groupby

from transformers import AutoTokenizer

from . import config

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def join_units(parts):
    text = parts[0]
    for part in parts[1:]:
        if re.search(r"[a-z]-$", text) and part[:1].islower():
            text = text[:-1] + part  # a word hyphenated across a page break
        else:
            text += "\n" + part
    return text


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    section: str
    page_start: int
    page_end: int
    page_label: str
    text: str
    tokens: int
    position: int  # order inside its document, used to fetch neighbours

    @property
    def context_text(self):
        # the book and section path give the embedder and BM25 context a bare paragraph lacks
        return f"{self.doc_title} | {self.section}\n{self.text}"


class Chunker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL)

    def count(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def units(self, block):
        """One unit per block, unless the block is too long, then sentence groups that fit."""
        n = self.count(block.text)
        if n <= config.CHUNK_TOKENS:
            return [(block.text, n, block)]

        out, current, current_n = [], [], 0
        for sentence in SENTENCE_END.split(block.text):
            s_n = self.count(sentence)
            if current and current_n + s_n > config.CHUNK_TOKENS:
                out.append((" ".join(current), current_n, block))
                current, current_n = [], 0
            current.append(sentence)
            current_n += s_n
        if current:
            out.append((" ".join(current), current_n, block))
        return out

    def overlap_from(self, unit):
        """Tail of the previous chunk carried into the next one, whole sentences only."""
        text, n, block = unit
        if n <= config.OVERLAP_TOKENS:
            return [unit]
        tail, tail_n = [], 0
        for sentence in reversed(SENTENCE_END.split(text)):
            s_n = self.count(sentence)
            if tail_n + s_n > config.OVERLAP_TOKENS:
                break
            tail.insert(0, sentence)
            tail_n += s_n
        return [(" ".join(tail), tail_n, block)] if tail else []

    def chunk_document(self, blocks, doc_id, doc_title):
        chunks = []

        def emit(units, section):
            if sum(u[1] for u in units) < 25:
                return  # chapter title pages: a heading and nothing else
            text = join_units([u[0] for u in units])
            chunks.append(Chunk(
                id=f"{doc_id}:{len(chunks)}",
                doc_id=doc_id,
                doc_title=doc_title,
                section=" > ".join(section[-3:]) or doc_title,
                page_start=units[0][2].page,
                page_end=units[-1][2].page,
                page_label=units[0][2].page_label,
                text=text,
                tokens=sum(u[1] for u in units),
                position=len(chunks),
            ))

        for section, group in groupby(blocks, key=lambda b: b.section):
            current, current_n, carried = [], 0, 0
            section_start = len(chunks)

            for block in group:
                # start a fresh chunk at a heading rather than burying it mid-chunk
                if block.is_heading and current_n >= config.MIN_CHUNK_TOKENS:
                    emit(current, section)
                    current, current_n, carried = [], 0, 0

                for unit in self.units(block):
                    if current and current_n + unit[1] > config.CHUNK_TOKENS:
                        emit(current, section)
                        current = self.overlap_from(current[-1])
                        current_n = sum(u[1] for u in current)
                        carried = len(current)
                    current.append(unit)
                    current_n += unit[1]

            fresh = current[carried:]  # the carried overlap is already in the previous chunk
            if not fresh:
                continue
            fresh_n = sum(u[1] for u in fresh)
            last = chunks[-1] if len(chunks) > section_start else None
            if current_n < config.MIN_CHUNK_TOKENS and last and last.tokens + fresh_n <= config.CHUNK_TOKENS + config.MIN_CHUNK_TOKENS:
                # tiny leftover at the end of a section reads better glued to the chunk before it
                last.text = join_units([last.text] + [u[0] for u in fresh])
                last.tokens += fresh_n
                last.page_end = fresh[-1][2].page
            else:
                emit(current, section)

        return chunks
