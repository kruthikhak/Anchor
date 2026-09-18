import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer

from . import config, prompts
from .index import STEMMER, STOPWORDS, embedder
from .llm import chat_json

WORD = re.compile(r"[a-z]+")
TOKEN = re.compile(r"[A-Za-z0-9]+")

# Placement acronyms. Where one means two different things in these books, the student is asked
# which one they meant rather than the system guessing, unless the chosen subject settles it.
ACRONYMS = {
    "dsa": [("Data Structures and Algorithms", "DSA"), ("Digital Signature Algorithm", "Computer Networks")],
    "fd": [("functional dependency", "DBMS"), ("file descriptor", "Operating Systems")],
    "bfs": [("breadth-first search", "DSA")],
    "dfs": [("depth-first search", "DSA")],
    "bst": [("binary search tree", "DSA")],
    "mst": [("minimum spanning tree", "DSA")],
    "dp": [("dynamic programming", "DSA")],
    "dag": [("directed acyclic graph", "DSA")],
    "lcs": [("longest common subsequence", "DSA")],
    "lis": [("longest increasing subsequence", "DSA")],
    "os": [("operating system", "Operating Systems")],
    "ipc": [("inter-process communication", "Operating Systems")],
    "tlb": [("translation lookaside buffer", "Operating Systems")],
    "lru": [("least recently used", "Operating Systems")],
    "acid": [("atomicity, consistency, isolation, durability", "DBMS")],
    "dbms": [("database management system", "DBMS")],
    "erd": [("entity relationship diagram", "DBMS")],
    "pk": [("primary key", "DBMS")],
    "fk": [("foreign key", "DBMS")],
    "ddl": [("data definition language", "DBMS")],
    "dml": [("data manipulation language", "DBMS")],
    "wal": [("write-ahead logging", "DBMS")],
    "2pl": [("two-phase locking", "DBMS")],
    "mvcc": [("multiversion concurrency control", "DBMS")],
    "tcp": [("transmission control protocol", "Computer Networks")],
    "udp": [("user datagram protocol", "Computer Networks")],
    "dns": [("domain name system", "Computer Networks")],
    "arp": [("address resolution protocol", "Computer Networks")],
    "nat": [("network address translation", "Computer Networks")],
    "dhcp": [("dynamic host configuration protocol", "Computer Networks")],
    "osi": [("open systems interconnection model", "Computer Networks")],
    "cidr": [("classless interdomain routing", "Computer Networks")],
}

# section titles too vague to offer as a topic on their own
VAGUE_TITLES = {"introduction", "summary", "overview", "perspective", "key terms", "the pattern", "applications",
                "analysis", "implementation", "example", "examples", "variants", "notes", "discussion"}


@dataclass
class Understood:
    query: str
    corrections: list = field(default_factory=list)  # (what they typed, what the books spell)
    expansions: list = field(default_factory=list)  # (acronym, meaning)
    clarify: dict = None  # set when an acronym has two meanings and nothing says which
    alternatives: list = field(default_factory=list)  # (acronym, meaning, subject) the subject ruled out


def clean_title(title):
    title = re.sub(r"^(chapter|appendix)\s+\w+\s+", "", title.strip(), flags=re.I)  # "Chapter 12 Normalization"
    title = re.sub(r"^\d+(\.\d+)*\s+", "", title)  # "5.2 Reliable Byte Stream", but not "2-4 Trees" or "2SAT"
    return title.strip(" ♥*⋆")


class QueryHelper:
    def __init__(self, chunks):
        counts = Counter(w for c in chunks for w in WORD.findall(c.text.lower()))
        self.vocabulary = {w for w, n in counts.items() if n >= 3}

        titles = {clean_title(part) for c in chunks for part in c.section.split(">")}
        self.titles = sorted(t for t in titles if len(t) > 3 and t.lower() not in VAGUE_TITLES)
        chapters = {clean_title(c.section.split(">")[0]) for c in chunks}
        self.chapters = sorted(t for t in chapters if len(t) > 3 and t.lower() not in VAGUE_TITLES)
        self._title_vectors = None

        # Spelling is only ever corrected towards a topic word, one that names a section somewhere, so
        # an off-syllabus term like "sharding" is never quietly turned into an unrelated "sharing".
        topic_stems = {STEMMER.stemWord(w) for t in self.titles for w in WORD.findall(t.lower()) if len(w) >= 4}
        topic_stems |= {STEMMER.stemWord(w) for ms in ACRONYMS.values() for m, _ in ms for w in WORD.findall(m.lower())}
        self.targets = defaultdict(list)
        for word in self.vocabulary:
            if STEMMER.stemWord(word) in topic_stems:
                self.targets[word[:2]].append(word)

        # every whole word in the embedder's vocabulary is ordinary English, and real words aren't typos
        self.english = {w for w in AutoTokenizer.from_pretrained(config.EMBED_MODEL).vocab if w.isalpha()}

    def unused_words(self, text):
        """Words the books never use: typos, slips like "mutation" for mutex, or topics we don't cover."""
        return [w for w in dict.fromkeys(WORD.findall(text.lower()))
                if len(w) >= 4 and w not in self.vocabulary and w not in STOPWORDS and w not in ACRONYMS]

    def unknown_words(self, text):
        """The unused words that aren't ordinary English either, so most likely typos."""
        return [w for w in self.unused_words(text) if w not in self.english]

    def closest_topic_word(self, word):
        # the fallback when the model can't be asked; typos rarely touch the first letters, which
        # is what stops "mutation" turning into "emulation"
        match = process.extractOne(word, self.targets.get(word[:2], []),
                                   scorer=Levenshtein.normalized_similarity, score_cutoff=0.8)
        return match[0] if match else word

    def spelling(self, text, words):
        # only a model can tell "dedlock" (a typo) from "sharding" (a real term these books don't use)
        try:
            fixes = chat_json(
                [
                    {"role": "system", "content": prompts.SPELLING_SYSTEM},
                    {"role": "user", "content": f"Question: {text}\nWords: {', '.join(words)}"},
                ],
                model=config.REWRITE_MODEL,
                reasoning_effort="low",
                temperature=0,
                max_completion_tokens=1200,
            ).get("words", {})
            return {w: str(fixes.get(w, w)) for w in words}
        except Exception:
            return {w: self.closest_topic_word(w) for w in words}

    def fix_spelling(self, text):
        unknown = self.unknown_words(text)
        if not unknown:
            return text, []
        corrections = []
        for word, meant in self.spelling(text, unknown).items():
            meant = re.sub(r"['’]s\b", "", meant.lower()).strip()  # "Kruskal's" searches the same as "kruskal"
            # only accept a spelling the books actually use, otherwise the search gains nothing
            words = [w for w in WORD.findall(meant) if len(w) > 2]
            if meant != word and words and all(w in self.vocabulary for w in words):
                corrections.append((word, meant))
                text = re.sub(rf"\b{re.escape(word)}\b", meant, text, flags=re.I)
        return text, corrections

    def understand(self, question, subject=None):
        text, corrections = self.fix_spelling(question)

        expansions, clarify, alternatives = [], None, []
        for token in dict.fromkeys(t.lower() for t in TOKEN.findall(text)):
            meanings = ACRONYMS.get(token)
            if not meanings or any(m.lower() in text.lower() for m, _ in meanings):
                continue  # not one we know, or the question already spells it out
            if len(meanings) > 1 and subject not in (None, "", "All"):
                chosen = [m for m in meanings if m[1] == subject]
                if chosen:
                    # still offered, in case the subject wasn't what they had in mind
                    alternatives += [(token.upper(), m, s) for m, s in meanings if (m, s) not in chosen]
                    meanings = chosen
            if len(meanings) > 1:
                clarify = {"term": token.upper(), "options": [m for m, _ in meanings]}
                continue
            meaning = meanings[0][0]
            expansions.append((token.upper(), meaning))
            # "OSI model" becomes "OSI (open systems interconnection) model", not "... model) model",
            # which read oddly enough to make the model refuse once
            shown = meaning
            after = re.search(rf"\b{re.escape(token)}\s+(\w+)", text, flags=re.I)
            if after and meaning.lower().endswith(" " + after.group(1).lower()):
                shown = meaning[: -len(after.group(1))].strip()
            text = re.sub(rf"\b{re.escape(token)}\b", f"{token.upper()} ({shown})", text, count=1, flags=re.I)

        return Understood(text, corrections, expansions, clarify, alternatives)

    def candidate_topics(self, query):
        if self._title_vectors is None:
            self._title_vectors = embedder().encode(self.titles, normalize_embeddings=True, batch_size=64)
        vector = embedder().encode([query], prompt=config.QUERY_INSTRUCTION, normalize_embeddings=True)[0]
        by_meaning = [self.titles[i] for i in np.argsort(-(self._title_vectors @ vector))[:12]]
        by_spelling = [t for t, _, _ in process.extract(query, self.titles, scorer=fuzz.WRatio, limit=6)]
        # Look-alikes first, since a list read top-down favours what comes early. The chapters go
        # last so that an idea no title shares words with, like the banker's algorithm, still has
        # somewhere to point (Synchronization and Deadlocks).
        return list(dict.fromkeys(self.lookalikes(query) + by_spelling + by_meaning + self.chapters))

    def lookalikes(self, query):
        """Titles with a word that starts like an ordinary English word the books never use."""
        # A half-remembered term often comes out as a real word with the same first letters:
        # "mutation" for mutex or mutual exclusion. A word that isn't English, like "sharding", has
        # already been through the spelling check, so what's left of those are real terms.
        closeness = {}
        for word in (w for w in self.unused_words(query) if w in self.english):
            for title in self.titles:
                for title_word in WORD.findall(title.lower()):
                    if len(title_word) >= 5 and title_word[:3] == word[:3]:
                        closeness[title] = max(closeness.get(title, 0), fuzz.ratio(word, title_word))
        return sorted(closeness, key=closeness.get, reverse=True)[:8]

    def suggest(self, query):
        """Which of the books' own section titles to offer when the library can't answer."""
        candidates = self.candidate_topics(query)
        request = (f"Question: {query}\n"
                   f"Words the textbooks never use: {', '.join(self.unused_words(query)) or 'none'}\n"
                   f"Titles with a word that starts like one of those: {'; '.join(self.lookalikes(query)) or 'none'}\n\n"
                   "Section titles:\n" + "\n".join(candidates))
        try:
            # the larger model: this only runs on a refusal, and the small one was too quick to say "unrelated"
            data = chat_json(
                [{"role": "system", "content": prompts.SUGGEST_SYSTEM}, {"role": "user", "content": request}],
                reasoning_effort="low",
                temperature=0,
                max_completion_tokens=1200,
            )
        except Exception:
            data = None
        if not isinstance(data, dict):
            # a failed call says nothing about the question, so don't claim that nothing is close
            return {"kind": "unavailable", "topics": []}

        picked = data.get("topics")
        picked = [picked] if isinstance(picked, str) else picked if isinstance(picked, list) else []
        topics = [t for t in picked if t in candidates][:3]
        kind = data.get("kind") if data.get("kind") in ("typo", "related") and topics else "unrelated"
        return {"kind": kind, "topics": topics if kind != "unrelated" else []}
