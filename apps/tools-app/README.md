# tools-app

## MinIO storage

A single local MinIO container backs storage for two pages, each in its own bucket:

- **Job Application** -- session files (`cv.tex`, `application.tex`, `job_posting.tex`), bucket `job-app`.
- **Transcribe** -- saved recordings + transcripts (`../transcribe/storage.py`), bucket `transcribe`.

```bash
cp .env.example .env   # then fill in MINIO_ROOT_USER / MINIO_ROOT_PASSWORD
docker compose up -d
uv sync --group dev
uv run streamlit run main.py
```

MinIO's web console is at http://localhost:9001 (sign in with the credentials from `.env`).

Transcribe can also run standalone (`cd ../transcribe && uv run streamlit run main.py`) against
this same container -- see `../transcribe/.env.example`.
