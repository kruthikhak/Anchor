"""Every endpoint, end to end on the real index, with a stand-in for the models so no API is called.

Needs the index from scripts/build_index.py, and skips itself without it."""
import json
import re
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from openai import BadRequestError, RateLimitError

from rag import config, llm, prompts

INDEX_BUILT = (config.INDEX_DIR / "chunks.jsonl").exists()
ANSWER = "A deadlock needs mutual exclusion, hold and wait, no preemption and a circular wait [1]."


def reply(content, model):
    return SimpleNamespace(model=model, choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])


def streamed(text, model):
    for word in text.split(" "):
        yield SimpleNamespace(model=model, choices=[SimpleNamespace(delta=SimpleNamespace(content=word + " "))])


def api_error(kind, status):
    return kind("refused", response=httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com")), body=None)


class FakeModels:
    """Replies to each kind of request the app sends the way a model would."""

    def __init__(self):
        self.chat = self.completions = self
        self.limited = set()  # models that answer with a 429
        self.suggest = {"kind": "unrelated", "topics": []}
        self.quiz = []
        self.broken_json = False

    def create(self, model, messages, stream=False, **kwargs):
        if model in self.limited:
            raise api_error(RateLimitError, 429)
        system = messages[0]["content"]
        if stream:
            return streamed(ANSWER, model)
        if system == prompts.REWRITE_SYSTEM:
            return reply("What is a semaphore used for?", model)
        if self.broken_json:
            raise api_error(BadRequestError, 400)
        if system == prompts.SPELLING_SYSTEM:
            return reply(json.dumps({"words": {}}), model)
        if system == prompts.SUGGEST_SYSTEM:
            return reply(json.dumps(self.suggest), model)
        if system == prompts.PRACTICE_SYSTEM:
            return reply(json.dumps({"questions": [
                {"question": "What four conditions make a deadlock?", "answer": "Mutual exclusion, hold and wait, no preemption, circular wait.",
                 "explanation": "All four have to hold at once.", "source": 1},
                {"question": "A question from nowhere?", "answer": "No source.", "source": None},
            ]}), model)
        if system == prompts.QUIZ_SYSTEM:
            return reply(json.dumps({"questions": self.quiz}), model)
        raise AssertionError(f"a request the app shouldn't make: {system[:60]}")


def events(response):
    return [(re.search(r"^event: (.+)$", part, re.M).group(1), json.loads(re.search(r"^data: (.+)$", part, re.M).group(1)))
            for part in response.text.strip().split("\n\n")]


@unittest.skipUnless(INDEX_BUILT, "the index isn't built")
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.server import app

        cls.models = FakeModels()
        cls.patch = mock.patch.object(llm, "client", lambda: cls.models)
        cls.patch.start()
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()  # runs the start-up that loads the index and the models

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        cls.patch.stop()

    def setUp(self):
        self.models.limited = set()
        self.models.suggest = {"kind": "unrelated", "topics": []}
        self.models.broken_json = False

    def ask(self, question, **extra):
        return dict(events(self.client.post("/api/ask", json={"question": question, **extra})))

    def test_an_answer_streams_its_sources_then_the_text(self):
        got = self.ask("What conditions must hold for a deadlock to occur?")
        self.assertTrue(got["sources"]["grounded"])
        self.assertEqual(len(got["sources"]["sources"]), 5)
        self.assertEqual(got["done"]["answer"].strip(), ANSWER)
        self.assertEqual(got["done"]["citations"], [1])
        self.assertFalse(got["done"]["refused"])
        self.assertFalse(got["done"]["fellBack"])

    def test_a_follow_up_is_searched_as_its_rewrite(self):
        history = [{"role": "user", "content": "What is a semaphore?"}, {"role": "assistant", "content": "A counter [1]."}]
        got = self.ask("what is it used for?", history=history)
        self.assertEqual(got["sources"]["searchQuery"], "What is a semaphore used for?")

    def test_an_ambiguous_acronym_is_asked_about_before_searching(self):
        got = self.ask("what is DSA")
        self.assertEqual([o["meaning"] for o in got["clarify"]["options"]], ["Data Structures and Algorithms", "Digital Signature Algorithm"])
        self.assertNotIn("sources", got)
        self.assertTrue(got["done"]["clarify"])

    def test_the_subject_settles_an_acronym_and_the_other_meaning_is_offered(self):
        got = self.ask("what is DSA", subject="DSA")
        self.assertEqual(got["sources"]["expansions"], [["DSA", "Data Structures and Algorithms"]])
        self.assertEqual(got["sources"]["alternatives"][0]["subject"], "Computer Networks")

    def test_off_syllabus_is_refused_without_calling_the_answer_model(self):
        got = self.ask("how do i bake sourdough bread")
        self.assertFalse(got["sources"]["grounded"])
        self.assertTrue(got["done"]["refused"])
        self.assertEqual(got["suggest"], {"kind": "unrelated", "topics": []})

    def test_a_suggestion_must_be_one_of_the_books_titles(self):
        self.models.suggest = {"kind": "typo", "topics": ["Mutexes and Monitors", "A title no book has"]}
        got = self.ask("mutation")
        self.assertEqual(got["suggest"], {"kind": "typo", "topics": ["Mutexes and Monitors"]})

    def test_the_fallback_model_is_used_and_reported(self):
        self.models.limited = {config.ANSWER_MODEL}
        got = self.ask("What conditions must hold for a deadlock to occur?")
        self.assertEqual(got["done"]["model"], config.FALLBACK_MODEL)
        self.assertTrue(got["done"]["fellBack"])

    def test_no_model_left_is_an_error_event_not_a_crash(self):
        self.models.limited = {config.ANSWER_MODEL, config.FALLBACK_MODEL, config.LAST_RESORT_MODEL}
        got = self.ask("What conditions must hold for a deadlock to occur?")
        self.assertIn("free-tier", got["error"]["message"])

    def test_an_empty_question_is_turned_away(self):
        self.assertIn("error", self.ask("   "))

    def test_practice_keeps_only_questions_that_name_a_source(self):
        data = self.client.post("/api/practice", json={"topic": "What conditions must hold for a deadlock to occur?"}).json()
        self.assertEqual([q["question"] for q in data["questions"]], ["What four conditions make a deadlock?"])
        self.assertEqual(data["questions"][0]["explanation"], "All four have to hold at once.")
        self.assertEqual(self.client.post("/api/practice", json={"topic": " "}).status_code, 422)

    def test_a_quiz_is_sized_by_its_section_and_checked(self):
        self.models.quiz = [
            {"type": "mcq", "question": "Which is not a deadlock condition?", "options": ["Mutual exclusion", "Hold and wait", "Preemption", "Circular wait"],
             "answer": "C", "explanation": "Deadlock needs no preemption.", "source": 1},
            {"type": "blank", "question": "Ordering locks prevents a circular ____.", "answer": "wait", "source": 2},
            {"type": "mcq", "question": "Broken: the answer isn't an option", "options": ["A", "B", "C"], "answer": 9, "source": 1},
        ]
        data = self.client.post("/api/quiz", json={"topic": "Explain Deadlock"}).json()
        self.assertEqual(data["marks"], 10)
        self.assertEqual([q["type"] for q in data["questions"]], ["mcq", "blank"])
        mcq = data["questions"][0]
        self.assertEqual(mcq["options"][mcq["answer"]], "Preemption")

    def test_a_quiz_on_an_ambiguous_acronym_asks_first(self):
        self.assertIn("clarify", self.client.post("/api/quiz", json={"topic": "DSA"}).json())

    def test_a_quiz_the_model_cannot_write_is_a_message_not_a_crash(self):
        self.models.broken_json = True
        response = self.client.post("/api/quiz", json={"topic": "Explain Deadlock"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("try again", response.json()["detail"])

    def test_topics_follow_the_books_and_their_subjects(self):
        books = {b["id"]: b for b in self.client.get("/api/topics").json()["books"]}
        self.assertEqual(len(books), 9)
        chapters = {c["title"]: c for c in books["hailperin-operating-systems"]["chapters"]}
        self.assertEqual(chapters["Atomic Transactions"]["subjects"], ["Operating Systems", "DBMS"])
        self.assertEqual(chapters["Virtual Memory"]["subjects"], ["Operating Systems"])

    def test_a_topic_page_lists_its_passages_in_order(self):
        data = self.client.get("/api/topic", params={"book": "hailperin-operating-systems", "path": "Synchronization and Deadlocks > Deadlock"}).json()
        self.assertEqual(data["total"], len(data["passages"]))
        self.assertTrue(any(p["skip"] for p in data["passages"]))
        missing = self.client.get("/api/topic", params={"book": "hailperin-operating-systems", "path": "Nothing Here"})
        self.assertEqual(missing.status_code, 404)

    def test_a_long_section_comes_a_page_at_a_time(self):
        tcp = {"book": "peterson-davie-computer-networks", "path": "End-to-End Protocols > Reliable Byte Stream (TCP)"}
        first = self.client.get("/api/topic", params=tcp).json()
        rest = self.client.get("/api/topic", params={**tcp, "start": len(first["passages"])}).json()
        self.assertEqual(len(first["passages"]) + len(rest["passages"]), first["total"])
        self.assertEqual(rest["passages"][0]["number"], len(first["passages"]) + 1)

    def test_search_finds_passages_and_handles_the_odd_cases(self):
        found = self.client.get("/api/search", params={"q": "semaphore"}).json()["passages"]
        self.assertTrue(any("Semaphores" in p["section"] for p in found))
        self.assertEqual(self.client.get("/api/search", params={"q": "  "}).json()["passages"], [])
        both = self.client.get("/api/search", params={"q": "fd"}).json()["passages"]
        self.assertTrue({"watt-database-design", "hailperin-operating-systems"} <= {p["docId"] for p in both})

    def test_library_and_evaluation_pages(self):
        self.assertEqual(len(self.client.get("/api/library").json()["books"]), 9)
        evaluation = self.client.get("/api/evaluation").json()
        self.assertIn("H", evaluation["retrieval"])
        self.assertEqual(evaluation["answers"]["answerable"], 39)


if __name__ == "__main__":
    unittest.main()
