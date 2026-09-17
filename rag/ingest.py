import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import pymupdf

# leaving out TEXT_PRESERVE_LIGATURES makes "ﬁrst" come out as "first", which matters for keyword search
TEXT_FLAGS = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_MEDIABOX_CLIP

# Index pages are keyword soup, and exercises read like questions but never contain answers,
# so both rank well for question-shaped queries while being useless as sources.
SKIP_SECTIONS = re.compile(
    r"^(index( of \w+)?|bibliography|references|further reading|notes|(table of )?contents"
    r"|acknowledg\w*|preface|about this book|about the exercises|steal this book|image credits|colophon"
    r"|exercises|discussion and exercises|programming projects|exploration projects)$",
    re.I,
)

NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)+\s+[A-Z]")  # "5.2.3 Connection Establishment"
WORD = re.compile(r"[A-Za-z]{3,}")


@dataclass
class Block:
    doc_id: str
    page: int  # physical page, 1-based, what a PDF viewer jumps to
    page_label: str  # number printed on the page, when the PDF defines one
    section: tuple
    text: str
    is_heading: bool


def clean(text):
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    # TEXT_DEHYPHENATE misses line-end hyphens followed by a stray space in some books
    text = re.sub(r"(?<=[a-z])-[ \t]*\n\s*(?=[a-z])", "", text)
    return " ".join(text.split())


def normalise_title(text):
    text = re.sub(r"^[\d.§]+\s*", "", text.strip())
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_furniture(pages, heights):
    """Running heads and page numbers sit at the same height on most pages of a book.
    Count single-line blocks near the top and bottom edge by position. A position that
    repeats on a quarter of the pages is furniture, but only if most blocks there carry a
    number: books without running heads also start the body text at a fixed height."""
    seen = {"top": Counter(), "bottom": Counter()}
    numbered = {"top": Counter(), "bottom": Counter()}
    for blocks, height in zip(pages, heights):
        for x0, y0, x1, y1, text, *_ in blocks:
            if y1 - y0 > 20 or len(text) > 100:
                continue
            if y0 < 0.15 * height:
                edge, pos = "top", round(y0 / 3)
            elif y1 > 0.85 * height:
                edge, pos = "bottom", round(y1 / 3)
            else:
                continue
            seen[edge][pos] += 1
            numbered[edge][pos] += any(ch.isdigit() for ch in text)

    needed = 0.25 * len(pages)
    return tuple(
        {pos for pos, n in seen[edge].items() if n >= needed and numbered[edge][pos] >= 0.6 * n}
        for edge in ("top", "bottom")
    )


def is_furniture(block, height, furniture):
    x0, y0, x1, y1, text, *_ = block
    if y1 - y0 > 20 or len(text) > 100:
        return False
    top, bottom = furniture
    if y0 < 0.15 * height and round(y0 / 3) in top:
        return True
    return y1 > 0.85 * height and round(y1 / 3) in bottom


def extract_blocks(path, doc_id):
    doc = pymupdf.open(path)
    pages = [
        [b for b in page.get_text("blocks", flags=TEXT_FLAGS, sort=True) if b[6] == 0]
        for page in doc
    ]
    heights = [page.rect.height for page in doc]
    furniture = find_furniture(pages, heights)

    toc = [(level, clean(title), page) for level, title, page in doc.get_toc() if page >= 1]
    starts = defaultdict(list)
    for level, title, page in toc:
        starts[page].append((level, title))

    # title page, copyright and table of contents all come before the first toc entry
    first_page = toc[0][2] if toc else 1

    section = []
    skipping = False
    out = []
    for pno, blocks in enumerate(pages, start=1):
        if pno < first_page:
            continue
        height = heights[pno - 1]
        texts = [clean(b[4]) for b in blocks if not is_furniture(b, height, furniture)]
        texts = [t for t in texts if t]

        # find the block where each section starting on this page begins,
        # if the heading text can't be matched assume it starts where the previous one did
        transitions = defaultdict(list)
        cursor = 0
        for level, title in starts.get(pno, []):
            key = normalise_title(title)
            match = next((i for i in range(cursor, len(texts)) if normalise_title(texts[i]).startswith(key)), cursor)
            transitions[match].append((level, title))
            cursor = match

        label = doc[pno - 1].get_label()
        for i, text in enumerate(texts):
            for level, title in transitions.get(i, []):
                section = section[: level - 1] + [title]
                skipping = False

            heading = i in transitions or bool(NUMBERED_HEADING.match(text))
            if not toc and len(text) < 80 and NUMBERED_HEADING.match(text):
                section = [text]  # lecture notes have no toc, numbered headings are the best we get
                skipping = False
            if len(text) < 40 and SKIP_SECTIONS.match(normalise_title(text)):
                # end-of-chapter exercises and notes that some PDFs leave out of their toc
                skipping = True

            if skipping or any(SKIP_SECTIONS.match(normalise_title(s)) for s in section):
                continue
            # stray labels and maths fragments from figures
            if not heading and len(WORD.findall(text)) < 2:
                continue
            out.append(Block(doc_id, pno, label, tuple(section), text, heading))

        # a section that starts on a page with nothing left after cleaning still needs to take effect
        for i in sorted(k for k in transitions if k >= len(texts)):
            for level, title in transitions[i]:
                section = section[: level - 1] + [title]
                skipping = False

    return out
