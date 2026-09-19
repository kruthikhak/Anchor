import json
import os
import re
from dataclasses import asdict
from functools import lru_cache

import faiss
import numpy as np
import snowballstemmer
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config
from .chunking import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STEMMER = snowballstemmer.stemmer("english")
# question words carry no topic and would get a high IDF because textbooks rarely use them
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or that the this "
    "to was what when where which who why will with you your explain describe tell give briefly please".split()
)


def device():
    # the Space sets this to cpu: on ZeroGPU hardware torch reports a GPU that only exists inside @spaces.GPU calls
    if os.environ.get("ANCHOR_DEVICE"):
        return os.environ["ANCHOR_DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder():
    return SentenceTransformer(config.EMBED_MODEL, device=device())


@lru_cache(maxsize=2)
def reranker(name=config.RERANK_MODEL):
    return CrossEncoder(name, device=device(), max_length=512)


def bm25_tokens(text):
    words = [w for w in TOKEN.findall(text.lower()) if w not in STOPWORDS]
    return STEMMER.stemWords(words)


def save_chunks(chunks, directory=config.INDEX_DIR):
    with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks():
    with open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def build_dense(chunks, variant, directory=config.INDEX_DIR):
    texts = [c.context_text if variant == "contextual" else c.text for c in chunks]
    vectors = embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    # vectors are unit length, so inner product is cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, str(directory / f"dense_{variant}.faiss"))
