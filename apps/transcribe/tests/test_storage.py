import io
from dataclasses import dataclass
from typing import BinaryIO

from minio.error import S3Error

from storage import RecordingStorage, new_recording_id


@dataclass
class _MockObject:
    object_name: str | None
    is_dir: bool


class _MockResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class _MockMinio:
    """In-memory mock implementing storage.MinioClient, for tests."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._buckets: set[str] = set()

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self._buckets

    def make_bucket(self, bucket_name: str) -> None:
        self._buckets.add(bucket_name)

    def list_objects(
        self, bucket_name: str, prefix: str | None = None, recursive: bool = False
    ) -> list[_MockObject]:
        prefix = prefix or ""
        keys = sorted(k for k in self._objects if k.startswith(prefix))
        if recursive:
            return [_MockObject(k, False) for k in keys]
        seen_dirs: set[str] = set()
        results = []
        for k in keys:
            rest = k[len(prefix):]
            if "/" in rest:
                full_dir = prefix + rest.split("/", 1)[0] + "/"
                if full_dir not in seen_dirs:
                    seen_dirs.add(full_dir)
                    results.append(_MockObject(full_dir, True))
            else:
                results.append(_MockObject(k, False))
        return results

    def get_object(self, bucket_name: str, object_name: str) -> _MockResponse:
        if object_name not in self._objects:
            raise S3Error(None, "NoSuchKey", "not found", object_name, "", "", bucket_name, object_name)  # type: ignore[arg-type]
        return _MockResponse(self._objects[object_name])

    def put_object(
        self, bucket_name: str, object_name: str, data: BinaryIO, length: int, content_type: str = ""
    ) -> None:
        self._objects[object_name] = data.read()


def _storage() -> RecordingStorage:
    return RecordingStorage(_MockMinio(), "transcribe")


def test_new_recording_id_is_sortable_and_unique_per_second() -> None:
    a = new_recording_id()
    assert len(a) == len("20260806-153000")


def test_save_then_load_roundtrips() -> None:
    storage = _storage()
    storage.save("20260806-120000", b"fake-audio-bytes", ".wav", "hello world", "en", "tiny")

    assert storage.load_transcript("20260806-120000") == "hello world"
    assert storage.load_meta("20260806-120000") == {"language": "en", "model": "tiny"}
    assert storage.audio_filename("20260806-120000") == "20260806-120000/audio.wav"
    assert storage.load_audio("20260806-120000") == b"fake-audio-bytes"


def test_list_recordings_sorted_newest_first() -> None:
    storage = _storage()
    storage.save("20260806-100000", b"a", ".wav", "a", "en", "tiny")
    storage.save("20260806-120000", b"b", ".wav", "b", "en", "tiny")
    storage.save("20260805-090000", b"c", ".wav", "c", "en", "tiny")

    assert storage.list_recordings() == ["20260806-120000", "20260806-100000", "20260805-090000"]


def test_load_audio_missing_recording_returns_none() -> None:
    storage = _storage()
    assert storage.load_audio("nonexistent") is None
