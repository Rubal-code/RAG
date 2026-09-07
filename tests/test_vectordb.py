import tempfile

from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore


def test_vector_store():
    """
    Check that documents can be stored
    and retrieved from ChromaDB.
    """

    # Use a temporary database for testing
    temp_dir = tempfile.mkdtemp()

    vector_store = VectorStore(
        path=temp_dir,
        collection_name="test_documents"
    )

    embedder = Embedder(
        model_name="all-MiniLM-L6-v2"
    )

    documents = [
        "Python is a programming language.",
        "Machine learning uses data to learn patterns.",
        "RAG combines retrieval with generation."
    ]

    embeddings = embedder.embed(documents)

    # Store documents
    vector_store.add_documents(
        documents,
        embeddings
    )

    # Create query embedding
    query_embedding = embedder.embed(
        ["What is RAG?"]
    )

    # Search
    results = vector_store.search(
        query_embedding,
        n_results=2
    )

    # Make sure results exist
    assert "documents" in results
    assert len(results["documents"]) > 0
    assert len(results["documents"][0]) == 2