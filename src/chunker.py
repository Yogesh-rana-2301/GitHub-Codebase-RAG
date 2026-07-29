from dataclasses import dataclass


@dataclass
class Chunk:
    """A single text chunk produced by a chunking strategy."""
    content:   str   # the actual text
    filepath:  str   # which file this came from
    chunk_id:  str   # unique id: "filepath::strategy_N"
    strategy:  str   # which strategy produced this chunk
    start_char: int  # character offset in the original file (approximate for semantic)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Fixed-size chunking (with optional overlap)
# ──────────────────────────────────────────────────────────────────────────────

def fixed_size_chunks(
    documents:  list[dict],
    chunk_size: int = 500,
    overlap:    int = 0,
) -> list[Chunk]:
    """
    Split each document into fixed-size character windows.

    overlap > 0  →  consecutive chunks share `overlap` characters.
    This ensures that answers sitting at a chunk boundary are not lost —
    they appear in at least one of the two overlapping chunks.

    Example with chunk_size=10, overlap=3:
      text = "0123456789ABCDE"
      chunks: "0123456789", "789ABCDE"
                             ^^^  overlap
    """
    strategy_label = f"fixed_{chunk_size}_overlap_{overlap}"
    chunks = []

    for doc in documents:
        text     = doc["content"]
        filepath = doc["filepath"]
        start    = 0
        idx      = 0

        while start < len(text):
            end        = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= 50:   # ignore tiny fragments
                chunks.append(Chunk(
                    content    = chunk_text,
                    filepath   = filepath,
                    chunk_id   = f"{filepath}::{strategy_label}_{idx}",
                    strategy   = strategy_label,
                    start_char = start,
                ))
                idx += 1

            step  = chunk_size - overlap   # how far to advance the window
            start += max(step, 1)          # guard against infinite loop if step<=0

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Semantic (paragraph-boundary) chunking
# ──────────────────────────────────────────────────────────────────────────────

def semantic_chunks(
    documents:      list[dict],
    max_chunk_size: int = 600,
) -> list[Chunk]:
    """
    Split at natural paragraph boundaries (\n\n), accumulating paragraphs
    until the chunk would exceed max_chunk_size.

    Why this is better for documentation:
      A paragraph typically covers one concept.  Fixed-size chunking can cut
      a paragraph in the middle, leaving an incomplete thought in each half.
      Semantic chunking keeps related sentences together, which improves
      retrieval accuracy — the model gets complete context.
    """
    chunks = []

    for doc in documents:
        text     = doc["content"]
        filepath = doc["filepath"]

        paragraphs    = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk = ""
        idx           = 0

        for para in paragraphs:
            candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para

            if len(candidate) <= max_chunk_size:
                current_chunk = candidate
            else:
                # Flush what we have accumulated so far
                if len(current_chunk) >= 50:
                    chunks.append(Chunk(
                        content    = current_chunk,
                        filepath   = filepath,
                        chunk_id   = f"{filepath}::semantic_{idx}",
                        strategy   = "semantic",
                        start_char = 0,   # approximate — not tracked for semantic
                    ))
                    idx += 1
                current_chunk = para   # start a new chunk with the current paragraph

        # Flush the last accumulated chunk
        if len(current_chunk) >= 50:
            chunks.append(Chunk(
                content    = current_chunk,
                filepath   = filepath,
                chunk_id   = f"{filepath}::semantic_{idx}",
                strategy   = "semantic",
                start_char = 0,
            ))

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

STRATEGIES = ["fixed_no_overlap", "fixed_with_overlap", "semantic"]


def get_chunks_by_strategy(documents: list[dict], strategy: str) -> list[Chunk]:
    """Return chunks produced by the named strategy."""
    if strategy == "fixed_no_overlap":
        return fixed_size_chunks(documents, chunk_size=500, overlap=0)
    elif strategy == "fixed_with_overlap":
        return fixed_size_chunks(documents, chunk_size=500, overlap=100)
    elif strategy == "semantic":
        return semantic_chunks(documents, max_chunk_size=600)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}"
        )
