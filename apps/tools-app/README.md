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
