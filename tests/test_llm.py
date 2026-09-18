import unittest
from unittest import mock

import httpx
from openai import RateLimitError

from rag import config, llm


class FakeClient:
    """Pretends to be the Groq client. Models in `limited` answer with a 429."""

    def __init__(self, limited):
        self.limited, self.tried, self.caps = limited, [], {}
        self.chat = self.completions = self

    def create(self, model, messages, **kwargs):
        self.tried.append(model)
        self.caps[model] = kwargs.get("max_completion_tokens")
        if model in self.limited:
            response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
            raise RateLimitError("rate limited", response=response, body=None)
        return model


def ask(model, limited):
    fake = FakeClient(limited)
    with mock.patch.object(llm, "client", lambda: fake):
        return llm.chat([], model=model), fake.tried


@unittest.skipIf(config.ANSWER_MODEL == config.FALLBACK_MODEL, "ANSWER_MODEL is overridden for development")
class FallbackTests(unittest.TestCase):
    def test_answer_model_is_used_when_it_has_tokens(self):
        self.assertEqual(ask(config.ANSWER_MODEL, set()), (config.ANSWER_MODEL, [config.ANSWER_MODEL]))

    def test_answers_fall_back_to_the_small_model_then_the_last_resort(self):
        reply, tried = ask(config.ANSWER_MODEL, {config.ANSWER_MODEL, config.FALLBACK_MODEL})
        self.assertEqual(reply, config.LAST_RESORT_MODEL)
        self.assertEqual(tried, [config.ANSWER_MODEL, config.FALLBACK_MODEL, config.LAST_RESORT_MODEL])

    def test_small_jobs_leave_the_answer_model_for_last(self):
        _, tried = ask(config.REWRITE_MODEL, {config.REWRITE_MODEL, config.LAST_RESORT_MODEL})
        self.assertEqual(tried, [config.REWRITE_MODEL, config.LAST_RESORT_MODEL, config.ANSWER_MODEL])

    def test_requests_to_qwen_stay_under_its_output_limit(self):
        fake = FakeClient({config.ANSWER_MODEL, config.FALLBACK_MODEL})
        with mock.patch.object(llm, "client", lambda: fake):
            llm.chat([], model=config.ANSWER_MODEL, max_completion_tokens=1500)
        self.assertEqual(fake.caps, {config.ANSWER_MODEL: 1500, config.FALLBACK_MODEL: 1500, config.LAST_RESORT_MODEL: 1000})

    def test_error_is_raised_when_every_model_is_out(self):
        with self.assertRaises(RateLimitError):
            ask(config.ANSWER_MODEL, {config.ANSWER_MODEL, config.FALLBACK_MODEL, config.LAST_RESORT_MODEL})


if __name__ == "__main__":
    unittest.main()
