import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "space"

FILES = ["data/corpus.json"]
FOLDERS = ["rag", "app", "data/index", "eval/results"]


def main():
    if not (ROOT / "data" / "index" / "chunks.jsonl").exists():
        raise SystemExit("build the index first: python scripts/build_index.py")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # a Space expects its entry point, requirements and README (with the settings header) at the top level
    shutil.copy(ROOT / "deploy" / "serve.py", OUT / "serve.py")
    shutil.copy(ROOT / "deploy" / "space-requirements.txt", OUT / "requirements.txt")
    shutil.copy(ROOT / "deploy" / "space-readme.md", OUT / "README.md")

    for name in FILES:
        (OUT / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, OUT / name)
    for name in FOLDERS:
        shutil.copytree(ROOT / name, OUT / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"Space folder ready at {OUT.relative_to(ROOT)} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
