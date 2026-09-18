import unittest

from rag.chunking import Chunk
from rag.query import QueryHelper, clean_title


def chunk(section, text):
    return Chunk(id=section, doc_id="book", doc_title="Book", section=section, page_start=1, page_end=1,
                 page_label="1", text=text, tokens=100, position=0)


# a tiny library: a word needs to appear three times to count as one the books use
CHUNKS = [
    chunk("Synchronization > Deadlock", "deadlock deadlock deadlock mutex mutex mutex kruskal kruskal kruskal"),
    chunk("Graphs > Breadth-First Search", "search search search queue queue queue"),
]


class TitleTests(unittest.TestCase):
    def test_numbering_is_dropped(self):
        self.assertEqual(clean_title("Chapter 11 Functional Dependencies"), "Functional Dependencies")
        self.assertEqual(clean_title("5.10 Analysis of Binary Probing ⋆"), "Analysis of Binary Probing")
        self.assertEqual(clean_title("9.2.1 Summation"), "Summation")

    def test_numbers_that_are_part_of_the_name_stay(self):
        self.assertEqual(clean_title("2-4 Trees"), "2-4 Trees")
        self.assertEqual(clean_title("2SAT problem"), "2SAT problem")
        self.assertEqual(clean_title("B-Trees"), "B-Trees")


class UnderstandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = QueryHelper(CHUNKS)

    def setUp(self):
        # stands in for the model, so these tests never call the API
        self.fixes = {}
        self.helper.spelling = lambda text, words: {w: self.fixes.get(w, w) for w in words}

    def test_ambiguous_acronym_asks_without_a_subject(self):
        understood = self.helper.understand("what is DSA")
        self.assertEqual(understood.clarify["term"], "DSA")
        self.assertEqual(len(understood.clarify["options"]), 2)

    def test_subject_settles_an_ambiguous_acronym(self):
        self.assertIn("Data Structures and Algorithms", self.helper.understand("what is DSA", "DSA").query)
        self.assertIn("Digital Signature Algorithm", self.helper.understand("what is dsa", "Computer Networks").query)

    def test_subject_that_matches_neither_meaning_still_asks(self):
        self.assertIsNotNone(self.helper.understand("what is an FD", "DSA").clarify)

    def test_meaning_already_given_is_not_asked_again(self):
        understood = self.helper.understand("what is DSA (meaning Data Structures and Algorithms)?")
        self.assertIsNone(understood.clarify)
        self.assertEqual(understood.expansions, [])

    def test_every_known_acronym_is_spelled_out(self):
        understood = self.helper.understand("TCP vs UDP")
        self.assertEqual([a for a, _ in understood.expansions], ["TCP", "UDP"])
        self.assertIn("transmission control protocol", understood.query)

    def test_typo_is_fixed_when_the_books_use_the_fix(self):
        self.fixes = {"dedlock": "deadlock"}
        understood = self.helper.understand("what is a dedlock")
        self.assertEqual(understood.query, "what is a deadlock")
        self.assertEqual(understood.corrections, [("dedlock", "deadlock")])

    def test_fix_the_books_never_use_is_ignored(self):
        self.fixes = {"sharding": "shardingg"}
        understood = self.helper.understand("what is sharding")
        self.assertEqual(understood.query, "what is sharding")
        self.assertEqual(understood.corrections, [])

    def test_empty_or_tiny_fix_is_ignored(self):
        for fix in ("", "an"):
            self.fixes = {"dedlock": fix}
            self.assertEqual(self.helper.understand("what is a dedlock").query, "what is a dedlock")

    def test_possessive_is_dropped_from_a_fix(self):
        self.fixes = {"kruskals": "Kruskal's"}
        self.assertEqual(self.helper.understand("kruskals algorithm").query, "kruskal algorithm")

    def test_ordinary_english_is_never_sent_for_spelling(self):
        self.assertEqual(self.helper.unknown_words("what happens when the queue is empty"), [])
        self.assertEqual(self.helper.unknown_words("explain dedlock"), ["dedlock"])


if __name__ == "__main__":
    unittest.main()
