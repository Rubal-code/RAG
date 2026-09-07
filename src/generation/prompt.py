def build_prompt(query, retrieved_documents):
    """
    Build the final RAG prompt.
    """

    # Combine retrieved chunks
    context = "\n\n".join(retrieved_documents)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context,
say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt