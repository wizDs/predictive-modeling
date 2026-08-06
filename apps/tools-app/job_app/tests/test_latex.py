import subprocess
from pathlib import Path

import pytest

from job_app import latex


def test_docker_available_reflects_shutil_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latex.shutil, "which", lambda name: "/usr/bin/docker")
    assert latex.docker_available() is True

    monkeypatch.setattr(latex.shutil, "which", lambda name: None)
    assert latex.docker_available() is False


def test_compile_to_pdf_raises_when_docker_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latex.shutil, "which", lambda name: None)
    with pytest.raises(latex.LatexCompileError, match="docker not found"):
        latex.compile_to_pdf("\\documentclass{article}", "cv.tex")


def test_compile_to_pdf_writes_pdf_next_to_the_mounted_workdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake `docker run` just drops a PDF where pdflatex would have, to isolate our
    subprocess wiring (workdir mount, filename plumbing) from an actual LaTeX toolchain."""
    monkeypatch.setattr(latex.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        # Strip only the trailing ":/work" (not split on every ":") since a Windows path
        # itself contains a drive-letter colon, e.g. "C:\...\tmp:/work".
        workdir = Path(cmd[cmd.index("-v") + 1].removesuffix(":/work"))
        (workdir / "cv.pdf").write_bytes(b"%PDF-1.4 fake")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(latex.subprocess, "run", fake_run)
    assert latex.compile_to_pdf("\\documentclass{article}", "cv.tex") == b"%PDF-1.4 fake"


def test_compile_to_pdf_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latex.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="! Undefined control sequence.", stderr="")

    monkeypatch.setattr(latex.subprocess, "run", fake_run)
    with pytest.raises(latex.LatexCompileError, match="Undefined control sequence"):
        latex.compile_to_pdf("\\documentclass{article}", "cv.tex")


def test_compile_to_pdf_raises_when_pdf_missing_despite_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latex.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(latex.subprocess, "run", fake_run)
    with pytest.raises(latex.LatexCompileError):
        latex.compile_to_pdf("\\documentclass{article}", "cv.tex")


def test_compile_to_pdf_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latex.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd, 90)

    monkeypatch.setattr(latex.subprocess, "run", fake_run)
    with pytest.raises(latex.LatexCompileError, match="timed out"):
        latex.compile_to_pdf("\\documentclass{article}", "cv.tex")
