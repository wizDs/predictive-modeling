"""LaTeX-to-PDF compilation via a throwaway texlive Docker container.

Compiling CVs/cover letters to PDF needs a full LaTeX toolchain, which is too large to bundle
as an app dependency. Instead we shell out to `docker run` against a texlive image on demand --
no persistent container, no compose service, just Docker acting as the "install a TeX
distribution for me" mechanism. This mirrors how apps/transcribe treats ffmpeg: check whether
the external tool is available and surface install instructions rather than vendoring it.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Full texlive/texlive is several GB; override with a smaller scheme (e.g. a "-basic" tag) if
# your CV/cover letter don't need the full package set.
_IMAGE = os.environ.get("LATEX_DOCKER_IMAGE", "texlive/texlive:latest")
_TIMEOUT_SECONDS = 90

DOCKER_MISSING_MESSAGE = (
    "**docker not found.** PDF compilation runs LaTeX inside a `texlive/texlive` Docker "
    "container rather than requiring a full local TeX install.\n\n"
    "Install Docker, then reload:\n"
    "- **Windows/macOS:** https://docs.docker.com/desktop/\n"
    "- **Linux:** your distro's `docker` / `docker-ce` package"
)


class LatexCompileError(Exception):
    """Raised when pdflatex fails or produces no PDF; carries the tail of its log as the message."""


def docker_available() -> bool:
    return shutil.which("docker") is not None


def compile_to_pdf(tex_source: str, filename: str) -> bytes:
    """Compile `tex_source` to PDF bytes using a throwaway texlive container.

    Runs pdflatex twice (a single pass leaves references/page numbers unresolved) against a
    temp dir bind-mounted into the container, then reads back the resulting PDF.
    """
    if not docker_available():
        raise LatexCompileError(DOCKER_MISSING_MESSAGE)

    stem = Path(filename).stem
    with tempfile.TemporaryDirectory(prefix="job_app_latex_") as tmp:
        workdir = Path(tmp)
        workdir.joinpath(filename).write_text(tex_source, encoding="utf-8")

        compile_cmd = (
            f"pdflatex -interaction=nonstopmode -halt-on-error {filename} && "
            f"pdflatex -interaction=nonstopmode -halt-on-error {filename}"
        )
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{workdir}:/work",
                    "-w", "/work",
                    _IMAGE,
                    "sh", "-c", compile_cmd,
                ],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise LatexCompileError(f"pdflatex timed out after {_TIMEOUT_SECONDS}s") from exc

        pdf_path = workdir / f"{stem}.pdf"
        if result.returncode != 0 or not pdf_path.exists():
            log_tail = (result.stdout or result.stderr or "").strip()[-4000:]
            raise LatexCompileError(log_tail or f"pdflatex exited {result.returncode} with no output")
        return pdf_path.read_bytes()
