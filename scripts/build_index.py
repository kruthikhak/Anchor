import json
import time

from rag import config
from rag.chunking import Chunker
from rag.index import build_dense, save_chunks
from rag.ingest import extract_blocks


def main():
    sources = json.loads(config.CORPUS_FILE.read_text())
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunker = Chunker()

    chunks = []
    for src in sources:
        path = config.RAW_DIR / f"{src['id']}.pdf"
        if not path.exists():
            print(f"skipping {src['id']}, PDF not downloaded yet")
            continue
        doc_chunks = chunker.chunk_document(extract_blocks(path, src["id"]), src["id"], src["title"])
        print(f"{src['id']}: {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)

    save_chunks(chunks)

    # two versions so the eval can measure what the section header adds
    for variant in ("plain", "contextual"):
        start = time.time()
        build_dense(chunks, variant)
        print(f"{variant}: embedded {len(chunks)} chunks in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
