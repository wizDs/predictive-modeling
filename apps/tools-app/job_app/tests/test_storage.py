import io
from dataclasses import dataclass
from typing import BinaryIO

from minio.error import S3Error

from job_app.storage import SessionStorage, object_key


@dataclass
class _FakeObject:
    object_name: str | None
    is_dir: bool


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class _FakeMinio:
    """In-memory stand-in implementing job_app.storage.MinioClient, for tests."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._buckets: set[str] = set()

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self._buckets

    def make_bucket(self, bucket_name: str) -> None:
        self._buckets.add(bucket_name)

    def list_objects(
        self, bucket_name: str, prefix: str | None = None, recursive: bool = False
    ) -> list[_FakeObject]:
        prefix = prefix or ""
        keys = sorted(k for k in self._objects if k.startswith(prefix))
        if recursive:
            return [_FakeObject(k, False) for k in keys]
        seen_dirs: set[str] = set()
        results = []
        for k in keys:
            rest = k[len(prefix):]
            if "/" in rest:
                full_dir = prefix + rest.split("/", 1)[0] + "/"
                if full_dir not in seen_dirs:
                    seen_dirs.add(full_dir)
                    results.append(_FakeObject(full_dir, True))
            else:
                results.append(_FakeObject(k, False))
        return results

    def get_object(self, bucket_name: str, object_name: str) -> _FakeResponse:
        if object_name not in self._objects:
            raise S3Error(None, "NoSuchKey", "not found", object_name, "", "", bucket_name, object_name)  # type: ignore[arg-type]
        return _FakeResponse(self._objects[object_name])

    def put_object(
        self, bucket_name: str, object_name: str, data: BinaryIO, length: int, content_type: str = ""
    ) -> None:
        self._objects[object_name] = data.read()


def _storage() -> SessionStorage:
    return SessionStorage(_FakeMinio(), "job-app")


def test_object_key() -> None:
    assert object_key("acme", "draft", "cv.tex") == "acme/draft/cv.tex"


def test_load_missing_returns_empty_string() -> None:
    storage = _storage()
    assert storage.load("acme", "draft", "cv.tex") == ""


def test_save_then_load_roundtrips() -> None:
    storage = _storage()
    storage.save("acme", "draft", "cv.tex", "hello world")
    assert storage.load("acme", "draft", "cv.tex") == "hello world"


def test_list_sessions_and_versions() -> None:
    storage = _storage()
    storage.save("acme", "draft", "cv.tex", "a")
    storage.save("acme", "final", "cv.tex", "b")
    storage.save("initech", "draft", "cv.tex", "c")

    assert storage.list_sessions() == ["acme", "initech"]
    assert storage.list_versions("acme") == ["draft", "final"]
    assert storage.list_versions("initech") == ["draft"]
    assert storage.list_versions("nonexistent") == []
