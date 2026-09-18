import unittest

from app.server import source_number
from rag import prompts
from rag.assistant import cited_numbers, is_refusal, tidy_citations
from rag.grounding import sentences


class RefusalTests(unittest.TestCase):
    def test_refusal_survives_small_changes(self):
        self.assertTrue(is_refusal(prompts.NOT_FOUND))
        self.assertTrue(is_refusal("I couldn’t find this in the study material.\n"))
        self.assertTrue(is_refusal("  i couldn't find this in the study material"))

    def test_partial_answer_is_not_a_refusal(self):
        self.assertFalse(is_refusal("The material doesn't cover this fully, but a deadlock needs four conditions [1]."))


class CitationTests(unittest.TestCase):
    def test_only_real_citations_count(self):
        answer = "A queue is FIFO [1]. It has two ends [2][3].\n```\nx = a[1]\n```\nSee [9]."
        self.assertEqual(cited_numbers(answer, 5), [1, 2, 3])

    def test_gpt_oss_citation_style_is_tidied(self):
        self.assertEqual(tidy_citations("A stack is LIFO 【2†L4-L9】."), "A stack is LIFO [2].")


class GroundingSentenceTests(unittest.TestCase):
    def test_questions_analogies_and_labels_are_not_claims(self):
        answer = "\n".join([
            "## Deadlock",
            "**Key points:**",
            "Analogy: two cars meeting on a one-lane bridge, each waiting for the other to reverse.",
            "What would happen if every thread held one lock and wanted another?",
            "- A deadlock needs threads to hold resources while waiting for more [1].",
        ])
        self.assertEqual(sentences(answer), ["A deadlock needs threads to hold resources while waiting for more [1]."])

    def test_code_is_not_checked(self):
        self.assertEqual(sentences("```\nwhile (true) { lock(a); lock(b); }\n```"), [])


class PracticeSourceTests(unittest.TestCase):
    def test_the_forms_the_model_uses_are_read(self):
        self.assertEqual(source_number(2, 5), 2)
        self.assertEqual(source_number("3", 5), 3)
        self.assertEqual(source_number([4], 5), 4)

    def test_missing_or_out_of_range_is_rejected(self):
        for value in (None, [], "x", 0, 6):
            self.assertIsNone(source_number(value, 5))


class PromptTests(unittest.TestCase):
    def test_explain_mode_uses_the_measured_prompt_word_for_word(self):
        self.assertIs(prompts.system_prompt("explain"), prompts.ANSWER_SYSTEM)

    def test_other_modes_only_swap_the_style(self):
        system = prompts.system_prompt("socratic")
        self.assertIn(prompts.MODES["socratic"], system)
        self.assertNotIn(prompts.EXPLAIN_STYLE, system)
        self.assertIn("Every sentence you write must be backed by a source you cite.", system)


if __name__ == "__main__":
    unittest.main()
