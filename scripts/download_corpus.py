import json
import ssl
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# python.org builds on macOS don't pick up the system certificates, certifi's bundle works everywhere
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as resp, open(part, "wb") as out:
        while block := resp.read(1 << 16):
            out.write(block)
    part.rename(dest)


def looks_like_pdf(path):
    # some hosts answer with an HTML error page and a 200, so check the magic bytes
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def main():
    sources = json.loads((ROOT / "data" / "corpus.json").read_text())
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for src in sources:
        dest = RAW_DIR / f"{src['id']}.pdf"
        if not dest.exists():
            if not src["url"]:
                print(f"missing  {dest.name}: {src['note']}")
                continue
            print(f"fetching {dest.name}")
            download(src["url"], dest)

        if looks_like_pdf(dest):
            print(f"ok       {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"BAD      {dest.name} is not a PDF, delete it and download by hand")


if __name__ == "__main__":
    main()
