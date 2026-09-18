import unittest

from rag.practice import check_question


def mcq(**changes):
    return {"type": "mcq", "question": "Which condition is not needed for deadlock?",
            "options": ["Mutual exclusion", "Hold and wait", "Preemption", "Circular wait"],
            "answer": 2, "explanation": "Deadlock needs no preemption.", "source": 1, **changes}


class QuizQuestionTests(unittest.TestCase):
    def test_mcq_answer_follows_its_option_through_the_shuffle(self):
        q = check_question(mcq(), 5)
        self.assertEqual(q["options"][q["answer"]], "Preemption")
        self.assertEqual(sorted(q["options"]), sorted(mcq()["options"]))

    def test_mcq_answer_given_as_a_letter_or_the_option_text(self):
        for answer in ("C", "c", "C)", "Preemption", "preemption", "2", [2]):
            q = check_question(mcq(answer=answer), 5)
            self.assertEqual(q["options"][q["answer"]], "Preemption")

    def test_options_given_as_an_object(self):
        options = {"A": "Mutual exclusion", "B": "Hold and wait", "C": "Preemption", "D": "Circular wait"}
        q = check_question(mcq(options=options, answer="C"), 5)
        self.assertEqual(q["options"][q["answer"]], "Preemption")

    def test_broken_mcqs_are_dropped(self):
        for broken in (mcq(answer=7), mcq(answer=True), mcq(answer="Z"), mcq(options=["Yes", "No"]),
                       mcq(options=["A", "A", "B", "C"]), mcq(source=9), mcq(source=None), mcq(question="")):
            self.assertIsNone(check_question(broken, 5))

    def test_blank_needs_exactly_one_gap(self):
        blank = {"type": "blank", "question": "A thread waiting on a lock it can never get is in a ____.", "answer": "deadlock", "source": 2}
        self.assertEqual(check_question(blank, 5)["question"].count("____"), 1)
        self.assertEqual(check_question({**blank, "question": "A thread stuck waiting is in a __."}, 5)["question"], "A thread stuck waiting is in a ____.")
        self.assertIsNone(check_question({**blank, "question": "No gap here."}, 5))
        self.assertIsNone(check_question({**blank, "question": "Two ____ and ____."}, 5))
        self.assertIsNone(check_question({**blank, "answer": "a very long answer that is really a sentence"}, 5))

    def test_short_answer_needs_an_answer(self):
        short = {"type": "short", "question": "Why is thrashing bad?", "answer": "The CPU waits on paging.", "source": 1}
        self.assertEqual(check_question(short, 5)["type"], "short")
        self.assertIsNone(check_question({**short, "answer": ""}, 5))

    def test_unknown_types_and_shapes_are_dropped(self):
        for broken in ({"type": "essay", "question": "Discuss.", "answer": "...", "source": 1}, "a string", None, []):
            self.assertIsNone(check_question(broken, 5))


if __name__ == "__main__":
    unittest.main()
