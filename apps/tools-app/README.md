# tools-app

## Job Application storage (MinIO)

The Job Application page stores session files (`cv.tex`, `application.tex`, `job_posting.tex`)
in a local MinIO container instead of on disk.

```bash
cp .env.example .env   # then fill in MINIO_ROOT_USER / MINIO_ROOT_PASSWORD
docker compose up -d
uv sync --group dev
uv run streamlit run main.py
```

MinIO's web console is at http://localhost:9001 (sign in with the credentials from `.env`).

## Job Application PDF compilation

The Viewer tab can compile a saved `cv.tex` / `application.tex` to PDF, for inline preview and
download. This needs the `docker` CLI on PATH (not the MinIO compose stack above) — compilation
shells out to a throwaway `texlive/texlive` container per compile (see `job_app/latex.py`)
rather than requiring a local TeX install. Nothing to start ahead of time: the first compile
pulls the image, which is several GB. If you don't need the full package set, point
`LATEX_DOCKER_IMAGE` (see `.env.example`) at a smaller scheme/tag.

If `docker` isn't on PATH, the Viewer tab shows install instructions instead of the compile
button.
