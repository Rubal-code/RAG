from src.chunking.chunker import create_chunks


def test_create_chunks():
    """
    Check that text is split into multiple chunks.
    """

    text = "This is a test document. " * 100

    chunks = create_chunks(
        text,
        chunk_size=100,
        overlap=20
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= 100
        assert len(chunk) > 0