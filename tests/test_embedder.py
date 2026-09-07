from src.embeddings.embedder import Embedder


def test_embedder():
    """
    Check that the embedding model generates
    the expected 384-dimensional embeddings.
    """

    embedder = Embedder(
        model_name="all-MiniLM-L6-v2"
    )

    embeddings = embedder.embed(
        ["Hello world"]
    )

    # One sentence → one embedding
    assert embeddings.shape[0] == 1

    # all-MiniLM-L6-v2 → 384 dimensions
    assert embeddings.shape[1] == 384