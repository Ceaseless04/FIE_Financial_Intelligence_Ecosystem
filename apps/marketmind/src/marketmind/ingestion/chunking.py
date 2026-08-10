"""Document chunking.

Fully deterministic — same input, same chunks, every time. That matters twice
over: embeddings stay stable across re-ingestion, and citations remain valid
because a chunk's character offsets keep resolving to the same span.

The chunker prefers to split on structure (headings, then paragraphs, then
sentences) before falling back to a hard character cut, so a chunk rarely ends
mid-sentence and retrieval sees coherent text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fie_common.errors import ValidationError
from marketmind.domain.documents import Chunk, Document, chunk_id_for

#: Markdown-style or numbered headings, used to tag chunks with their section.
_HEADING_PATTERN = re.compile(
    r"^(?:#{1,6}\s+(?P<hash>.+)|(?P<numbered>(?:PART|ITEM)\s+[IVX0-9]+[.:]?\s+.+))$",
    re.MULTILINE | re.IGNORECASE,
)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk sizing, in characters.

    Characters rather than tokens because chunking must be deterministic and
    tokenizer-independent: the same document must chunk identically whether it
    is later embedded by Ollama, vLLM, or anything else.
    """

    max_chars: int = 1200
    #: Overlap carries context across a boundary so a fact split across two
    #: chunks is still retrievable from at least one of them.
    overlap_chars: int = 150
    #: Chunks shorter than this are merged into the previous one rather than
    #: stored — a 20-character fragment is noise in a similarity search.
    min_chars: int = 100

    def __post_init__(self) -> None:
        if self.max_chars < 50:
            raise ValueError("max_chars must be at least 50")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        if self.min_chars < 0 or self.min_chars > self.max_chars:
            raise ValueError("min_chars must be between 0 and max_chars")


@dataclass(frozen=True)
class _Section:
    """A span of the document with the heading it sits under."""

    start: int
    end: int
    heading: str | None


def _find_sections(content: str) -> list[_Section]:
    """Split on headings, keeping absolute offsets intact."""
    matches = list(_HEADING_PATTERN.finditer(content))
    if not matches:
        return [_Section(start=0, end=len(content), heading=None)]

    sections: list[_Section] = []
    if matches[0].start() > 0:
        sections.append(_Section(start=0, end=matches[0].start(), heading=None))

    for position, match in enumerate(matches):
        heading = (match.group("hash") or match.group("numbered") or "").strip()
        body_start = match.end()
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(content)
        if body_end > body_start:
            sections.append(_Section(start=body_start, end=body_end, heading=heading))
    return sections


def _split_points(text: str, limit: int) -> int:
    """Best split offset at or before ``limit``, preferring natural boundaries."""
    window = text[:limit]

    paragraph_breaks = list(_PARAGRAPH_BREAK.finditer(window))
    if paragraph_breaks:
        return paragraph_breaks[-1].end()

    sentence_breaks = list(_SENTENCE_END.finditer(window))
    if sentence_breaks:
        return sentence_breaks[-1].end()

    last_space = window.rfind(" ")
    if last_space > limit // 2:
        return last_space + 1

    # No natural boundary — a hard cut is better than an unbounded chunk.
    return limit


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Split a document into overlapping, offset-preserving chunks.

    Raises:
        ValidationError: if the document has no usable content.
    """
    config = config or ChunkingConfig()
    content = document.content
    if not content.strip():
        raise ValidationError(
            "cannot chunk a document with no content", details={"document_id": document.id}
        )

    chunks: list[Chunk] = []
    index = 0

    for section in _find_sections(content):
        cursor = section.start
        # Skip leading whitespace so a chunk never starts with a newline run.
        while cursor < section.end and content[cursor].isspace():
            cursor += 1

        while cursor < section.end:
            remaining = section.end - cursor
            if remaining <= config.max_chars:
                end = section.end
            else:
                relative_end = _split_points(content[cursor : section.end], config.max_chars)
                end = cursor + max(relative_end, config.min_chars or 1)

            text = content[cursor:end]
            stripped = text.strip()

            if stripped:
                # Trim whitespace from the span itself so the stored offsets
                # still resolve exactly to the stored text.
                leading = len(text) - len(text.lstrip())
                trailing = len(text) - len(text.rstrip())
                start_char = cursor + leading
                end_char = end - trailing

                if end_char - start_char >= config.min_chars or not chunks:
                    chunks.append(
                        Chunk(
                            id=chunk_id_for(document.id, index),
                            document_id=document.id,
                            index=index,
                            text=content[start_char:end_char],
                            start_char=start_char,
                            end_char=end_char,
                            section=section.heading,
                        )
                    )
                    index += 1
                elif chunks and chunks[-1].section == section.heading:
                    # Absorb an undersized tail into the previous chunk rather
                    # than storing a fragment that will never retrieve well.
                    previous = chunks[-1]
                    chunks[-1] = previous.model_copy(
                        update={
                            "text": content[previous.start_char : end_char],
                            "end_char": end_char,
                        }
                    )

            if end >= section.end:
                break

            advance = max(end - config.overlap_chars, cursor + 1)
            cursor = advance

    if not chunks:
        raise ValidationError("chunking produced no chunks", details={"document_id": document.id})
    return chunks


def verify_chunk_offsets(document: Document, chunks: list[Chunk]) -> bool:
    """Confirm every chunk's offsets still resolve to its stored text.

    Cheap invariant, run after ingestion: if it ever fails, every citation
    derived from these chunks points somewhere other than it claims.
    """
    return all(chunk.resolve_in(document) == chunk.text for chunk in chunks)


__all__ = ["ChunkingConfig", "chunk_document", "verify_chunk_offsets"]
