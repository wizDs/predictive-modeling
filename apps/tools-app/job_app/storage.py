"""MinIO-backed storage for job-app session files.

Session files (cv.tex, application.tex, job_posting.tex) are stored as objects under
keys of the form ``<session>/<version>/<filename>`` in a single bucket, replacing the
old local-disk layout at ``job_app/data/<session>/<version>/``.
"""

import io
import os
from collections.abc import Iterable
from typing import BinaryIO, Protocol

from minio import Minio
from minio.error import S3Error

FILENAMES = ("cv.tex", "application.tex", "job_posting.tex")


class _StorageObject(Protocol):
    object_name: str | None
    is_dir: bool


class _ObjectResponse(Protocol):
    def read(self) -> bytes: ...
    def close(self) -> None: ...
    def release_conn(self) -> None: ...


class MinioClient(Protocol):
    """The subset of minio.Minio's interface SessionStorage relies on, for testability."""

    def bucket_exists(self, bucket_name: str) -> bool: ...
    def make_bucket(self, bucket_name: str) -> object: ...
    def list_objects(
        self, bucket_name: str, prefix: str | None = None, recursive: bool = False
    ) -> Iterable[_StorageObject]: ...
    def get_object(self, bucket_name: str, object_name: str) -> _ObjectResponse: ...
    def put_object(
        self, bucket_name: str, object_name: str, data: BinaryIO, length: int, content_type: str = ...
    ) -> object: ...


def object_key(session: str, version: str, filename: str) -> str:
    return f"{session}/{version}/{filename}"


def _dir_names(object_names: Iterable[str], prefix_len: int) -> list[str]:
    """Strip a known prefix length and trailing '/' off S3 common-prefix object names."""
    return sorted({name[prefix_len:].rstrip("/") for name in object_names})


class SessionStorage:
    """Session/version file storage backed by an S3-compatible (MinIO) bucket."""

    def __init__(self, client: MinioClient, bucket: str) -> None:
        self._client = client
        self._bucket = bucket
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

    def list_sessions(self) -> list[str]:
        objects = self._client.list_objects(self._bucket, recursive=False)
        return _dir_names((o.object_name for o in objects if o.is_dir and o.object_name), 0)

    def list_versions(self, session: str) -> list[str]:
        prefix = f"{session}/"
        objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=False)
        return _dir_names((o.object_name for o in objects if o.is_dir and o.object_name), len(prefix))

    def load(self, session: str, version: str, filename: str) -> str:
        key = object_key(session, version, filename)
        try:
            response = self._client.get_object(self._bucket, key)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return ""
            raise
        try:
            return response.read().decode("utf-8")
        finally:
            response.close()
            response.release_conn()

    def save(self, session: str, version: str, filename: str, content: str) -> None:
        key = object_key(session, version, filename)
        data = content.encode("utf-8")
        self._client.put_object(
            self._bucket, key, io.BytesIO(data), length=len(data), content_type="text/plain; charset=utf-8"
        )


def client_from_env() -> Minio:
    """Build a Minio client from MINIO_* environment variables (see .env.example)."""
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.environ["MINIO_ROOT_USER"]
    secret_key = os.environ["MINIO_ROOT_PASSWORD"]
    secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def bucket_from_env() -> str:
    return os.environ.get("MINIO_BUCKET", "job-app")
