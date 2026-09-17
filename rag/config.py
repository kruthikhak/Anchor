from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
CORPUS_FILE = DATA_DIR / "corpus.json"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

# bge was trained with this instruction on queries only, passages go in as they are
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# bge truncates at 512 tokens, leave room for the section header we prepend
CHUNK_TOKENS = 380
OVERLAP_TOKENS = 60
MIN_CHUNK_TOKENS = 80

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ANSWER_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
REWRITE_MODEL = "openai/gpt-oss-20b"
