from collections import defaultdict

from .query import clean_title

TOPIC_PASSAGES = 60  # at a time; a longer section or chapter loads the rest as the reader asks


def page(chunk):
    return chunk.page_label or str(chunk.page_start)


def page_range(chunks):
    first, last = page(chunks[0]), page(chunks[-1])
    return first if first == last else f"{first}–{last}"


def parts_of(chunk):
    return [p.strip() for p in chunk.section.split(">")]


def table_of_contents(chunks, books):
    """Each book's chapters and their sections, in the order the book itself gives them."""
    by_book = defaultdict(list)
    for chunk in chunks:
        by_book[chunk.doc_id].append(chunk)

    contents = []
    for book in books:
        mine = sorted(by_book.get(book["id"], []), key=lambda c: c.position)
        if not mine:
            continue
        paths = [parts_of(c) for c in mine]
        # A book split into a few parts, like the handbook's I, II and III, is browsed by the chapters
        # inside them instead; three entries at the top would say nothing about what's in the book.
        # Parts give themselves away by holding most of the text three levels down.
        skip = 1 if len({p[0] for p in paths}) <= 3 and sum(len(p) > 2 for p in paths) > len(paths) / 2 else 0

        chapters = {}
        for chunk, parts in zip(mine, paths):
            level = min(skip, len(parts) - 1)
            chapter_path = " > ".join(parts[:level + 1])
            chapters.setdefault(chapter_path, {"title": clean_title(parts[level]), "path": chapter_path, "chunks": [], "sections": {}})
            chapters[chapter_path]["chunks"].append(chunk)
            if len(parts) > level + 1:
                section_path = " > ".join(parts[:level + 2])
                section = chapters[chapter_path]["sections"].setdefault(
                    section_path, {"title": clean_title(parts[level + 1]), "path": section_path, "chunks": []})
                section["chunks"].append(chunk)

        # a book can serve a second subject with only some of its chapters, like the OS book's
        # chapter on transactions for DBMS
        partial = book.get("subject_chapters", {})
        subjects_of = lambda title: [s for s in book["subjects"] if s not in partial or title in partial[s]]

        contents.append({
            "id": book["id"],
            "title": book["title"],
            "subjects": book["subjects"],
            "chapters": [
                {
                    "title": ch["title"], "path": ch["path"], "pages": page_range(ch["chunks"]), "passages": len(ch["chunks"]),
                    "subjects": subjects_of(ch["title"]),
                    "sections": [{"title": s["title"], "path": s["path"], "pages": page_range(s["chunks"]), "passages": len(s["chunks"])}
                                 for s in ch["sections"].values()],
                }
                for ch in chapters.values()
            ],
        })
    return contents


def overlap(previous, current):
    """How many characters at the start of a chunk repeat the end of the one before it."""
    for size in range(min(len(previous), len(current), 800), 20, -1):
        if previous.endswith(current[:size]):
            # the line break after the repeated part would otherwise open the passage with a blank line
            return size + len(current[size:]) - len(current[size:].lstrip())
    return 0


def topic_passages(chunks, book_id, path, start=0):
    """The chunks under a chapter or section in reading order, a page's worth from `start`."""
    found = sorted((c for c in chunks if c.doc_id == book_id and (c.section == path or c.section.startswith(path + " > "))),
                   key=lambda c: c.position)
    shown = found[start:start + TOPIC_PASSAGES]
    # neighbouring chunks share a few sentences so nothing is lost at a boundary; when they're read
    # one after another, the repeated start of each is hidden, including across two pages' worth
    before = [found[start - 1]] if 0 < start <= len(found) else [None]
    skips = [overlap(a.text, b.text) if a and b.position == a.position + 1 else 0 for a, b in zip(before + shown, shown)]
    return found, list(zip(shown, skips))
