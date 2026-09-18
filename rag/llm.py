import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

from . import config

load_dotenv(config.ROOT / ".env")


@lru_cache(maxsize=1)
def client():
    # Groq speaks the OpenAI API, so the official SDK works with a different base URL
    return OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=config.GROQ_BASE_URL, max_retries=2)


def chat(messages, model=config.ANSWER_MODEL, **kwargs):
    # Each model has its own free-tier allowance (8k tokens a minute, 200k a day). When the one asked
    # for runs out the others take over in turn, the small ones first so that the answer model's
    # tokens are left for answers.
    order = list(dict.fromkeys([model, config.FALLBACK_MODEL, config.LAST_RESORT_MODEL, config.ANSWER_MODEL]))
    for name in order:
        try:
            return client().chat.completions.create(model=name, messages=messages, **kwargs)
        except RateLimitError:
            if name == order[-1]:
                raise


def chat_json(messages, **kwargs):
    # Now and then gpt-oss spends its whole token allowance reasoning and replies with nothing,
    # which Groq's JSON mode turns into a 400. Asking a second time almost always works.
    for attempt in range(2):
        try:
            reply = chat(messages, response_format={"type": "json_object"}, **kwargs)
            return json.loads(reply.choices[0].message.content or "")
        except (BadRequestError, json.JSONDecodeError):
            if attempt:
                raise
