from src.ingestion.loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore


# 1. Load PDF
text = load_pdf("data/sample.pdf")

# 2. Create chunks
chunks = create_chunks(text)

print("Number of chunks:", len(chunks))


# 3. Create embeddings
embedder = Embedder()

embeddings = embedder.embed(chunks)

print("Embedding shape:", embeddings.shape)


# 4. Create vector store
vector_store = VectorStore()


# 5. Store chunks + embeddings
vector_store.add_documents(
    chunks,
    embeddings
)

print("Documents added to ChromaDB!")


# 6. Create query
query = "What is Bobby Kumar's educational qualification?"

query_embedding = embedder.embed([query])


# 7. Search ChromaDB
results = vector_store.search(
    query_embedding,
    n_results=3
)


# 8. Display results
print("\n===== SEARCH RESULTS =====")

for i, document in enumerate(results["documents"][0]):
    print(f"\n--- Result {i + 1} ---")
    print(document)