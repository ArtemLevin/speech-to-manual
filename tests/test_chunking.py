from speech_to_manual.services.chunking import ChunkPolicy, TextChunker


def test_chunking_with_overlap_tail() -> None:
    text = "\n".join(["A" * 30, "B" * 30, "C" * 30])
    policy = ChunkPolicy(chunk_char_limit=50, overlap_chars=10)

    chunks = TextChunker.split(text, policy)

    assert len(chunks) >= 2
    assert chunks[0]
    assert chunks[1]


def test_chunking_empty_text() -> None:
    policy = ChunkPolicy(chunk_char_limit=100, overlap_chars=10)
    assert TextChunker.split("   ", policy) == []
