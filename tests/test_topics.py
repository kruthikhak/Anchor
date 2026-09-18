import unittest

from rag.chunking import Chunk
from rag.topics import overlap, table_of_contents, topic_passages


def chunk(position, section, text, page=1):
    return Chunk(id=f"c{position}", doc_id="book", doc_title="Book", section=section, page_start=page, page_end=page,
                 page_label=str(page), text=text, tokens=100, position=position)


BOOK = [{"id": "book", "title": "Book", "subjects": ["DSA"]}]


class ContentsTests(unittest.TestCase):
    def test_chapters_and_sections_keep_the_book_order(self):
        chunks = [chunk(0, "Sorting > Quicksort", "a", 3), chunk(1, "Sorting > Mergesort", "b", 5),
                  chunk(2, "Sorting > Mergesort > Analysis", "c", 6), chunk(3, "Chapter 4 Graphs", "d", 9)]
        chapters = table_of_contents(chunks, BOOK)[0]["chapters"]
        self.assertEqual([c["title"] for c in chapters], ["Sorting", "Graphs"])
        self.assertEqual([(s["title"], s["pages"], s["passages"]) for s in chapters[0]["sections"]],
                         [("Quicksort", "3", 1), ("Mergesort", "5–6", 2)])
        self.assertEqual(chapters[1]["sections"], [])

    def test_a_book_in_a_few_parts_is_browsed_by_its_chapters(self):
        chunks = [chunk(0, "I Basics > Sorting > Quicksort", "a"), chunk(1, "I Basics > Sorting > Mergesort", "b"),
                  chunk(2, "II Graphs > Paths > Dijkstra", "c")]
        chapters = table_of_contents(chunks, BOOK)[0]["chapters"]
        self.assertEqual([c["title"] for c in chapters], ["Sorting", "Paths"])
        self.assertEqual(chapters[1]["path"], "II Graphs > Paths")

    def test_a_book_without_passages_is_left_out(self):
        self.assertEqual(table_of_contents([], BOOK), [])


class PassageTests(unittest.TestCase):
    def test_repeated_start_of_the_next_chunk_is_measured(self):
        shared = "and this sentence is carried over into the next chunk."
        self.assertEqual(overlap("First part " + shared, shared + " New part."), len(shared) + 1)
        self.assertEqual(overlap("Nothing in common here at all.", "Completely different text follows."), 0)
        # the line break after the repeat is skipped with it
        self.assertEqual(overlap("First part " + shared, shared + "\n  New part."), len(shared) + 3)

    def test_a_section_includes_its_subsections_but_not_its_neighbours(self):
        chunks = [chunk(0, "Sorting > Mergesort", "a" * 30), chunk(1, "Sorting > Mergesort > Analysis", "b" * 30),
                  chunk(2, "Sorting > Mergesorts", "c" * 30), chunk(3, "Sorting > Quicksort", "d" * 30)]
        found, shown = topic_passages(chunks, "book", "Sorting > Mergesort")
        self.assertEqual([c.position for c in found], [0, 1])
        self.assertEqual(len(shown), 2)


if __name__ == "__main__":
    unittest.main()
