from src.ingestion.loader import load_pdf
from src.chunking.chunker import create_chunks

text = load_pdf("data/sample.pdf")

chunks = create_chunks(text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} ---")
    print(chunk)