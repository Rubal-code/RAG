from src.config import load_config

from src.ingestion.loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.generation.prompt import build_prompt
from src.generation.llm import LLM


# ============================================================
# LOAD CONFIG
# ============================================================

config = load_config()

print("Configuration loaded successfully!")


# ============================================================
# 1. LOAD PDF
# ============================================================

text = load_pdf(config["pdf_path"])

print("PDF loaded successfully!")


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

chunks = create_chunks(
    text,
    chunk_size=config["chunking"]["chunk_size"],
    overlap=config["chunking"]["overlap"]
)

print("Number of chunks:", len(chunks))


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embedder = Embedder(
    model_name=config["embedding"]["model"]
)

embeddings = embedder.embed(chunks)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 4. CREATE VECTOR STORE
# ============================================================

vector_store = VectorStore(
    path=config["vector_db"]["path"],
    collection_name=config["vector_db"]["collection_name"]
)


# ============================================================
# 5. STORE DOCUMENTS
# ============================================================

vector_store.add_documents(
    chunks,
    embeddings
)

print("Documents upserted to ChromaDB!")


# ============================================================
# 6. QUERY
# ============================================================

query = "What technologies and skills does Bobby Kumar have?"

print("\n===== QUERY =====")
print(query)


# ============================================================
# 7. EMBED QUERY
# ============================================================

query_embedding = embedder.embed([query])

print("\nQuery embedding shape:", query_embedding.shape)


# ============================================================
# 8. RETRIEVE
# ============================================================

results = vector_store.search(
    query_embedding,
    n_results=config["retrieval"]["top_k"]
)


# ============================================================
# 9. GET RETRIEVED DOCUMENTS
# ============================================================

print("\n===== SEARCH RESULTS =====")

retrieved_documents = []

for i, document in enumerate(results["documents"][0]):

    print(f"\n--- Result {i + 1} ---")
    print(document)

    retrieved_documents.append(document)


# ============================================================
# 10. BUILD PROMPT
# ============================================================

prompt = build_prompt(
    query,
    retrieved_documents
)

print("\n===== AUGMENTED PROMPT =====")
print(prompt)


# ============================================================
# 11. CREATE LLM
# ============================================================

llm = LLM(
    model=config["generation"]["model"],
    provider=config["generation"]["provider"],
    max_tokens=config["generation"]["max_tokens"],
    temperature=config["generation"]["temperature"]
)


# ============================================================
# 12. GENERATE ANSWER
# ============================================================

answer = llm.generate(prompt)


# ============================================================
# 13. FINAL ANSWER
# ============================================================

print("\n===== FINAL ANSWER =====")
print(answer)