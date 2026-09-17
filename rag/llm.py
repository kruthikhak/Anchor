import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from . import config

load_dotenv(config.ROOT / ".env")


@lru_cache(maxsize=1)
def client():
    # Groq speaks the OpenAI API, so the official SDK works with a different base URL
    return OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=config.GROQ_BASE_URL, max_retries=2)


def chat(messages, model=config.ANSWER_MODEL, **kwargs):
    try:
        return client().chat.completions.create(model=model, messages=messages, **kwargs)
    except RateLimitError:
        if model == config.FALLBACK_MODEL:
            raise
        # free tier is 8k tokens a minute per model, the smaller model has its own allowance
        return client().chat.completions.create(model=config.FALLBACK_MODEL, messages=messages, **kwargs)
