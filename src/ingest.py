import os
import subprocess
import pathlib


def clone_repo(repo_url: str, target_dir: str = "data/repo") -> str:
    """
    Clone a GitHub repository into target_dir.
    Uses --depth=1 (shallow clone) to only fetch the latest commit — much faster.
    Skips cloning if the directory already exists.
    """
    if os.path.exists(target_dir):
        print(f"Repo already exists at '{target_dir}' — skipping clone.")
        return target_dir

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    print(f"Cloning {repo_url} → {target_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth=1", repo_url, target_dir],
        check=True
    )
    print("Clone complete.")
    return target_dir


def extract_documents(repo_dir: str) -> list[dict]:
    """
    Walk the cloned repo and extract text from:
      - .md  files (documentation, README, guides)
      - .py  files (source code with docstrings and comments)

    Returns a list of dicts:
      {
        "content":  str,   # full file text
        "filepath": str,   # path relative to repo root
        "filetype": str    # "markdown" or "python"
      }

    Files shorter than 100 characters are skipped — they carry no useful information.
    """
    documents = []
    repo_path = pathlib.Path(repo_dir)

    # ── Markdown files ──────────────────────────────────────────────────────────
    for md_file in sorted(repo_path.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
            if len(content) >= 100:
                documents.append({
                    "content":  content,
                    "filepath": str(md_file.relative_to(repo_path)),
                    "filetype": "markdown",
                })
        except Exception as exc:
            print(f"  [skip] {md_file}: {exc}")

    # ── Python files ─────────────────────────────────────────────────────────────
    for py_file in sorted(repo_path.rglob("*.py")):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore").strip()
            if len(content) >= 100:
                documents.append({
                    "content":  content,
                    "filepath": str(py_file.relative_to(repo_path)),
                    "filetype": "python",
                })
        except Exception as exc:
            print(f"  [skip] {py_file}: {exc}")

    print(f"Extracted {len(documents)} documents from '{repo_dir}'.")
    return documents
