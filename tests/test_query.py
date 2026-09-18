import unittest
from unittest import mock

from rag import query
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

    def test_the_meaning_the_subject_ruled_out_is_still_offered(self):
        understood = self.helper.understand("what is DSA", "DSA")
        self.assertEqual(understood.alternatives, [("DSA", "Digital Signature Algorithm", "Computer Networks")])
        self.assertEqual(self.helper.understand("TCP vs UDP", "DSA").alternatives, [])  # one meaning, nothing to offer

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

    def test_expansion_does_not_repeat_the_word_after_it(self):
        self.assertEqual(self.helper.understand("the layers of the OSI model").query,
                         "the layers of the OSI (open systems interconnection) model")
        self.assertEqual(self.helper.understand("is TCP protocol reliable").query,
                         "is TCP (transmission control) protocol reliable")
        self.assertEqual(self.helper.understand("what is OSI").query, "what is OSI (open systems interconnection model)")

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


class SuggestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = QueryHelper(CHUNKS)
        cls.helper.candidate_topics = lambda query: ["Deadlock", "Breadth-First Search"]

    def suggest(self, reply):
        with mock.patch.object(query, "chat_json", lambda *args, **kwargs: reply):
            return self.helper.suggest("dedlock")

    def test_only_titles_from_the_list_are_offered(self):
        self.assertEqual(self.suggest({"kind": "typo", "topics": ["Deadlock", "Made Up Title"]}),
                         {"kind": "typo", "topics": ["Deadlock"]})
        self.assertEqual(self.suggest({"kind": "typo", "topics": "Deadlock"}), {"kind": "typo", "topics": ["Deadlock"]})

    def test_nothing_usable_means_unrelated(self):
        for reply in ({"kind": "typo", "topics": []}, {"kind": "guess", "topics": ["Deadlock"]}, {"topics": 3}, {}):
            self.assertEqual(self.suggest(reply), {"kind": "unrelated", "topics": []})

    def test_a_failed_call_is_not_reported_as_unrelated(self):
        self.assertEqual(self.suggest(["not", "an", "object"])["kind"], "unavailable")
        with mock.patch.object(query, "chat_json", mock.Mock(side_effect=RuntimeError("no network"))):
            self.assertEqual(self.helper.suggest("dedlock")["kind"], "unavailable")

    def test_a_lone_word_still_gets_its_lookalikes(self):
        with mock.patch.object(query, "chat_json", lambda *args, **kwargs: {"kind": "unrelated", "topics": []}):
            self.assertEqual(self.helper.suggest("deadline"), {"kind": "lookalike", "topics": ["Deadlock"]})
            # a question has enough context for the model's judgement to stand
            self.assertEqual(self.helper.suggest("when is the deadline"), {"kind": "unrelated", "topics": []})

    def test_lookalikes_come_from_real_words_the_books_never_use(self):
        self.assertEqual(self.helper.lookalikes("what is a deadline"), ["Deadlock"])
        self.assertEqual(self.helper.lookalikes("what is a deadlock"), [])  # a word the books use needs none
        # words that aren't English are typos or real terms, and the spelling check has seen them already
        self.assertEqual(self.helper.lookalikes("what is a dedlock"), [])


if __name__ == "__main__":
    unittest.main()
