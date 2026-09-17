import json
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

# questions whose best passage reranks below this are refused without calling the LLM;
# eval/calibrate_refusal.py writes the value, 0 means everything goes to the LLM
_threshold_file = ROOT / "eval" / "results" / "refusal_threshold.json"
REFUSAL_THRESHOLD = json.loads(_threshold_file.read_text())["threshold"] if _threshold_file.exists() else 0.0

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ANSWER_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
REWRITE_MODEL = "openai/gpt-oss-20b"
