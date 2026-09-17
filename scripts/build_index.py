import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so this runs from any folder

from rag import config
from rag.chunking import Chunker
from rag.index import build_dense, save_chunks
from rag.ingest import extract_blocks


def main():
    sources = json.loads(config.CORPUS_FILE.read_text())
    chunker = Chunker()

    chunks = []
    for src in sources:
        path = config.RAW_DIR / f"{src['id']}.pdf"
        if not path.exists():
            print(f"skipping {src['id']}, PDF not downloaded yet")
            continue
        blocks = extract_blocks(path, src["id"], src.get("toc_fix"))
        doc_chunks = chunker.chunk_document(blocks, src["id"], src["title"])
        print(f"{src['id']}: {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)

    # build into a separate folder and swap the files in at the end, so the app never
    # reads new chunks against old vectors and a failed build leaves the old index intact
    staging = config.DATA_DIR / "index_building"
    staging.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, staging)

    # two versions so the eval can measure what the section header adds
    for variant in ("plain", "contextual"):
        start = time.time()
        build_dense(chunks, variant, staging)
        print(f"{variant}: embedded {len(chunks)} chunks in {time.time() - start:.0f}s")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for path in staging.iterdir():
        path.replace(config.INDEX_DIR / path.name)
    staging.rmdir()


if __name__ == "__main__":
    main()
