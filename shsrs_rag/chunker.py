"""
shsrs_rag/chunker.py
====================
Document chunking for TXT and Markdown files.
Uses token-aware sliding window chunking.
"""

from __future__ import annotations
import re
import tiktoken
from dataclasses import dataclass
from shsrs_rag.config import settings


@dataclass
class Chunk:
    text:        str
    chunk_index: int
    token_count: int
    char_start:  int
    char_end:    int


# ── Tokenizer ─────────────────────────────────────────────────────────────────
def _get_tokenizer():
    """Use cl100k_base (GPT-4 tokenizer) as a general token counter."""
    return tiktoken.get_encoding("cl100k_base")


# ── Text cleaning ─────────────────────────────────────────────────────────────
def _clean_markdown(text: str) -> str:
    """Strip markdown syntax while preserving structure."""
    # Remove code blocks — keep content
    text = re.sub(r'```[\w]*\n(.*?)```', r'\1', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Convert links to text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalise whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _clean_txt(text: str) -> str:
    """Basic cleanup for plain text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Sentence splitter ─────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for clean chunk boundaries."""
    # Split on sentence endings followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Also split on paragraph breaks
    result = []
    for s in sentences:
        parts = s.split('\n\n')
        result.extend(p.strip() for p in parts if p.strip())
    return result


# ── Core chunker ──────────────────────────────────────────────────────────────
def chunk_text(
    text:         str,
    chunk_size:   int = None,
    chunk_overlap: int = None,
    is_markdown:  bool = False,
) -> list[Chunk]:
    """
    Chunk text into overlapping windows of ~chunk_size tokens.

    Parameters
    ----------
    text          : raw document text
    chunk_size    : target tokens per chunk (default from settings)
    chunk_overlap : overlap tokens between chunks (default from settings)
    is_markdown   : if True, strip markdown syntax first

    Returns
    -------
    List of Chunk objects with text, index, token count, char positions
    """
    chunk_size    = chunk_size    or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Clean text
    if is_markdown:
        text = _clean_markdown(text)
    else:
        text = _clean_txt(text)

    if not text.strip():
        return []

    tokenizer  = _get_tokenizer()
    sentences  = _split_sentences(text)

    if not sentences:
        return []

    # Tokenize all sentences
    sentence_tokens = [tokenizer.encode(s) for s in sentences]

    chunks       = []
    chunk_idx    = 0
    i            = 0   # sentence pointer
    char_pos     = 0

    while i < len(sentences):
        current_tokens = []
        current_sents  = []
        current_chars  = []

        j = i
        while j < len(sentences):
            s_tokens = sentence_tokens[j]
            if current_tokens and len(current_tokens) + len(s_tokens) > chunk_size:
                break
            current_tokens.extend(s_tokens)
            current_sents.append(sentences[j])
            current_chars.append(len(sentences[j]))
            j += 1

        if not current_sents:
            # Single sentence exceeds chunk_size — take it anyway
            current_sents  = [sentences[i]]
            current_tokens = sentence_tokens[i]
            j = i + 1

        chunk_text_str = ' '.join(current_sents)
        char_start     = text.find(current_sents[0], char_pos)
        char_end       = char_start + len(chunk_text_str)

        chunks.append(Chunk(
            text        = chunk_text_str,
            chunk_index = chunk_idx,
            token_count = len(current_tokens),
            char_start  = max(0, char_start),
            char_end    = char_end,
        ))
        chunk_idx += 1

        # Advance with overlap — step back overlap_tokens worth of sentences
        overlap_tokens = 0
        step_back      = 0
        for k in range(j - 1, i - 1, -1):
            overlap_tokens += len(sentence_tokens[k])
            if overlap_tokens >= chunk_overlap:
                break
            step_back += 1

        i = max(i + 1, j - step_back)
        char_pos = max(0, char_start)

    return chunks


# ── File loader ───────────────────────────────────────────────────────────────
def load_and_chunk(
    content:   bytes | str,
    filename:  str,
    chunk_size:    int = None,
    chunk_overlap: int = None,
) -> list[Chunk]:
    """
    Load a TXT or Markdown file and return chunks.

    Parameters
    ----------
    content    : raw file bytes or string
    filename   : original filename (used to detect .md)
    """
    if isinstance(content, bytes):
        text = content.decode('utf-8', errors='replace')
    else:
        text = content

    is_markdown = filename.lower().endswith(('.md', '.markdown'))

    return chunk_text(
        text          = text,
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
        is_markdown   = is_markdown,
    )
