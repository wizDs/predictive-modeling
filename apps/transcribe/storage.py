"""MinIO-backed storage for saved recordings and transcripts.

Reuses the same MinIO container job_app already runs (see apps/tools-app/docker-compose.yml)
under a separate bucket, rather than standing up a second container. Each recording is stored
as a few objects under a shared ``<recording_id>/`` prefix: ``audio<ext>``, ``transcript.txt``,
and ``meta.txt``.
"""

import io
import os
from collections.abc import Iterable
from datetime import datetime
from typing import BinaryIO, Protocol

from minio import Minio
from minio.error import S3Error


class _StorageObject(Protocol):
    object_name: str | None
    is_dir: bool


class _ObjectResponse(Protocol):
    def read(self) -> bytes: ...
    def close(self) -> None: ...
    def release_conn(self) -> None: ...


class MinioClient(Protocol):
    """The subset of minio.Minio's interface RecordingStorage relies on, for testability."""

    def bucket_exists(self, bucket_name: str) -> bool: ...
    def make_bucket(self, bucket_name: str) -> object: ...
    def list_objects(
        self, bucket_name: str, prefix: str | None = None, recursive: bool = False
    ) -> Iterable[_StorageObject]: ...
    def get_object(self, bucket_name: str, object_name: str) -> _ObjectResponse: ...
    def put_object(
        self, bucket_name: str, object_name: str, data: BinaryIO, length: int, content_type: str = ...
    ) -> object: ...


def new_recording_id() -> str:
    """A sortable, human-readable id for a new recording, unique to the second."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class RecordingStorage:
    """Recording/transcript storage backed by an S3-compatible (MinIO) bucket."""

    def __init__(self, client: MinioClient, bucket: str) -> None:
        self._client = client
        self._bucket = bucket
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

    def list_recordings(self) -> list[str]:
        objects = self._client.list_objects(self._bucket, recursive=False)
        ids = {o.object_name.rstrip("/") for o in objects if o.is_dir and o.object_name}
        return sorted(ids, reverse=True)  # newest first -- ids are timestamp-sortable

    def save(
        self, recording_id: str, audio_bytes: bytes, audio_ext: str, transcript: str, language: str, model: str
    ) -> None:
        self._put(f"{recording_id}/audio{audio_ext}", audio_bytes, "application/octet-stream")
        self._put(f"{recording_id}/transcript.txt", transcript.encode("utf-8"), "text/plain; charset=utf-8")
        meta = f"language={language}\nmodel={model}\n"
        self._put(f"{recording_id}/meta.txt", meta.encode("utf-8"), "text/plain; charset=utf-8")

    def load_transcript(self, recording_id: str) -> str:
        return self._get(f"{recording_id}/transcript.txt").decode("utf-8")

    def load_meta(self, recording_id: str) -> dict[str, str]:
        raw = self._get(f"{recording_id}/meta.txt").decode("utf-8")
        return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)

    def audio_filename(self, recording_id: str) -> str | None:
        prefix = f"{recording_id}/audio"
        for obj in self._client.list_objects(self._bucket, prefix=prefix, recursive=True):
            if obj.object_name:
                return obj.object_name
        return None

    def load_audio(self, recording_id: str) -> bytes | None:
        filename = self.audio_filename(recording_id)
        if filename is None:
            return None
        return self._get(filename)

    def _put(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(
            self._bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )

    def _get(self, key: str) -> bytes:
        try:
            response = self._client.get_object(self._bucket, key)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return b""
            raise
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


def client_from_env() -> Minio:
    """Build a Minio client from MINIO_* environment variables (see .env.example).

    Points at the same MinIO container job_app uses (apps/tools-app/docker-compose.yml) --
    only the bucket differs, so the two apps share one container instead of running two.
    """
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.environ["MINIO_ROOT_USER"]
    secret_key = os.environ["MINIO_ROOT_PASSWORD"]
    secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def bucket_from_env() -> str:
    return os.environ.get("TRANSCRIBE_MINIO_BUCKET", "transcribe")
