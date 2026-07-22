import subprocess
from pathlib import Path

DOCS_URL = "https://github.com/fastapi/fastapi"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = PROJECT_ROOT / "data" / "fastapi_docs"
SPARSE_PATHS = ["docs/en/docs", "docs/en/mkdocs.yml"]

def fetch_fastapi_docs():
    """
    Fetches the FastAPI documentation from the specified URL and saves it to the destination directory.
    """
    if DEST_DIR.exists():
        print(f"Destination directory '{DEST_DIR}' already exists. Pulling latest.", flush=True)
        subprocess.run(["git", "pull"], cwd=DEST_DIR, check=True)
        subprocess.run(["git", "sparse-checkout", "set", *SPARSE_PATHS], cwd=DEST_DIR, check=True)
        return
    else:
        print(f"Cloning FastAPI documentation from '{DOCS_URL}' to '{DEST_DIR}'...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", DOCS_URL, str(DEST_DIR)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", *SPARSE_PATHS], cwd=DEST_DIR, check=True)


if __name__ == "__main__":
    fetch_fastapi_docs()