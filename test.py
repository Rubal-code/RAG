from src.ingestion.loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.generation.prompt import build_prompt
from src.generation.llm import LLM


# ============================================================
# 1. LOAD PDF
# ============================================================

text = load_pdf("data/sample.pdf")

print("PDF loaded successfully!")


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

chunks = create_chunks(text)

print("Number of chunks:", len(chunks))


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embedder = Embedder()

embeddings = embedder.embed(chunks)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 4. CREATE VECTOR STORE
# ============================================================

vector_store = VectorStore()


# ============================================================
# 5. STORE DOCUMENTS
# ============================================================

vector_store.add_documents(
    chunks,
    embeddings
)

print("Documents added to ChromaDB!")


# ============================================================
# 6. USER QUESTION
# ============================================================

query = "What technologies and skills does Bobby Kumar have?"

print("\n===== QUERY =====")
print(query)


# ============================================================
# 7. EMBED USER QUESTION
# ============================================================

query_embedding = embedder.embed([query])

print("\nQuery embedding shape:", query_embedding.shape)


# ============================================================
# 8. RETRIEVE RELEVANT CHUNKS
# ============================================================

results = vector_store.search(
    query_embedding,
    n_results=3
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
# 10. BUILD AUGMENTED PROMPT
# ============================================================

prompt = build_prompt(
    query,
    retrieved_documents
)


print("\n===== AUGMENTED PROMPT =====")
print(prompt)


# ============================================================
# 11. GENERATE ANSWER
# ============================================================

llm = LLM()

answer = llm.generate(prompt)


# ============================================================
# 12. DISPLAY FINAL ANSWER
# ============================================================

print("\n===== FINAL ANSWER =====")
print(answer)