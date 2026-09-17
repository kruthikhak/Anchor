from collections import defaultdict
from dataclasses import dataclass, field

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .chunking import Chunk
from .index import bm25_tokens, embedder, load_chunks, reranker

# k from the original RRF paper, stops the top one or two ranks from drowning out the rest
RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    ranks: dict = field(default_factory=dict)  # position in each ranking, None if that retriever missed it
    rerank_score: float = None


def reciprocal_rank_fusion(rankings):
    # rank-based, so BM25 scores and cosine similarities never need to be put on the same scale
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] += 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    # section headers made fusion worse and the reranker no better (eval/results/retrieval_summary.txt),
    # so passages are indexed as plain text by default
    def __init__(self, contextual=False):
        self.contextual = contextual
        self.chunks = load_chunks()
        variant = "contextual" if contextual else "plain"
        self.dense = faiss.read_index(str(config.INDEX_DIR / f"dense_{variant}.faiss"))
        self.bm25 = BM25Okapi([bm25_tokens(self.passage(c)) for c in self.chunks])
        self.positions = {(c.doc_id, c.position): i for i, c in enumerate(self.chunks)}

    def passage(self, chunk):
        return chunk.context_text if self.contextual else chunk.text

    def dense_ranking(self, query, k):
        vector = embedder().encode([query], prompt=config.QUERY_INSTRUCTION, normalize_embeddings=True)
        _, ids = self.dense.search(vector.astype(np.float32), k)
        return [int(i) for i in ids[0] if i >= 0]

    def bm25_ranking(self, query, k):
        scores = self.bm25.get_scores(bm25_tokens(query))
        top = np.argpartition(-scores, k)[:k]
        return [int(i) for i in top[np.argsort(-scores[top])] if scores[i] > 0]

    # a pool of 20 scored the same as 30 in the eval and reranks about 40% faster on CPU
    def search(self, query, method="hybrid", rerank=True, k=5, pool=20, rerank_model=config.RERANK_MODEL, docs=None):
        depth = pool if docs is None else pool * 5  # narrowing to one subject throws candidates away
        rankings = {}
        if method in ("dense", "hybrid"):
            rankings["dense"] = self.dense_ranking(query, depth)
        if method in ("bm25", "hybrid"):
            rankings["bm25"] = self.bm25_ranking(query, depth)
        if docs is not None:
            rankings = {name: [i for i in r if self.chunks[i].doc_id in docs] for name, r in rankings.items()}
        rankings = {name: r[:pool] for name, r in rankings.items()}
        positions = {name: {idx: rank for rank, idx in enumerate(r, start=1)} for name, r in rankings.items()}

        if rerank:
            # Each retriever nominates its own top candidates. Cutting the pool by RRF order instead
            # drops chunks only one retriever found, even BM25's first result, in favour of chunks
            # that both ranked as mediocre.
            share = pool // len(rankings)
            candidates = list(dict.fromkeys(i for r in rankings.values() for i in r[:share]))
        else:
            candidates = reciprocal_rank_fusion(rankings.values())[:pool]
        hits = [Hit(self.chunks[i], {name: p.get(i) for name, p in positions.items()}) for i in candidates]

        if rerank and hits:
            # the cross-encoder reads query and passage together, much sharper than comparing two vectors
            scores = reranker(rerank_model).predict([(query, self.passage(h.chunk)) for h in hits], batch_size=16)
            for hit, score in zip(hits, scores):
                hit.rerank_score = float(score)
            hits.sort(key=lambda h: h.rerank_score, reverse=True)

        return hits[:k]

    def next_chunk(self, chunk):
        i = self.positions.get((chunk.doc_id, chunk.position + 1))
        return self.chunks[i] if i is not None else None
