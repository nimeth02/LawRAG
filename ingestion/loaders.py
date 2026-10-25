"""Bytes -> plain text, per file type. Add a loader here to support a new format."""
from __future__ import annotations

import io
import re
from pathlib import Path


class UnsupportedFileError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def normalise_text(text: str) -> str:
    """Unify line endings, strip trailing spaces, collapse runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_plain(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _load_pdf(data: bytes) -> str:
    from pypdf import PdfReader  # imported lazily: only needed for PDFs

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n\n".join(p for p in pages if p)


LOADERS = {
    ".txt": _load_plain,
    ".md": _load_plain,
    ".pdf": _load_pdf,
}


def supported_extensions() -> tuple[str, ...]:
    return tuple(LOADERS)


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileError(f"unsupported file type '{ext or filename}'; allowed: {', '.join(LOADERS)}")
    try:
        text = normalise_text(loader(data))
    except (UnsupportedFileError, EmptyDocumentError):
        raise
    except Exception as exc:  # noqa: BLE001 - parser failures of any kind
        raise UnsupportedFileError(f"could not read {filename}: {exc}") from exc
    if not text:
        raise EmptyDocumentError(f"no extractable text in {filename} (scanned PDF?)")
    return text
