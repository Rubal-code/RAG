from src.config import load_config

from src.ingestion.loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.generation.prompt import build_prompt
from src.generation.llm import LLM


class RAGPipeline:

    def __init__(self):
        """
        Initialize the complete RAG pipeline.
        """

        # Load configuration
        self.config = load_config()

        # ----------------------------------------------------
        # Create embedding model
        # ----------------------------------------------------
        self.embedder = Embedder(
            model_name=self.config["embedding"]["model"]
        )

        # ----------------------------------------------------
        # Create vector database
        # ----------------------------------------------------
        self.vector_store = VectorStore(
            path=self.config["vector_db"]["path"],
            collection_name=self.config["vector_db"]["collection_name"]
        )

        # ----------------------------------------------------
        # Create LLM
        # ----------------------------------------------------
        self.llm = LLM(
            model=self.config["generation"]["model"],
            provider=self.config["generation"]["provider"],
            max_tokens=self.config["generation"]["max_tokens"],
            temperature=self.config["generation"]["temperature"]
        )


    def ingest(self):
        """
        Load the PDF, create chunks, create embeddings,
        and store everything in ChromaDB.
        """

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------
        text = load_pdf(
            self.config["pdf_path"]
        )

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------
        chunks = create_chunks(
            text,
            chunk_size=self.config["chunking"]["chunk_size"],
            overlap=self.config["chunking"]["overlap"]
        )

        # ----------------------------------------------------
        # Create embeddings
        # ----------------------------------------------------
        embeddings = self.embedder.embed(chunks)

        # ----------------------------------------------------
        # Store in ChromaDB
        # ----------------------------------------------------
        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        print(f"Ingested {len(chunks)} chunks successfully.")


    def ask(self, question):
        """
        Answer a question using the RAG pipeline.
        """

        # ----------------------------------------------------
        # Convert question into an embedding
        # ----------------------------------------------------
        query_embedding = self.embedder.embed(
            [question]
        )

        # ----------------------------------------------------
        # Retrieve relevant chunks
        # ----------------------------------------------------
        results = self.vector_store.search(
            query_embedding,
            n_results=self.config["retrieval"]["top_k"]
        )

        # Get retrieved documents
        retrieved_documents = results["documents"][0]

        # ----------------------------------------------------
        # Build RAG prompt
        # ----------------------------------------------------
        prompt = build_prompt(
            question,
            retrieved_documents
        )

        # ----------------------------------------------------
        # Generate final answer
        # ----------------------------------------------------
        answer = self.llm.generate(
            prompt
        )

        return answer


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":

    # Create RAG pipeline
    rag = RAGPipeline()

    # Ingest the PDF
    rag.ingest()

    # Ask a question
    question = "What technologies and skills does Bobby Kumar have?"

    # Get answer
    answer = rag.ask(question)

    print("\n===== FINAL ANSWER =====")
    print(answer)