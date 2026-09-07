def build_prompt(query, retrieved_documents):
    """
    Build a prompt using the user's question
    and the chunks retrieved from ChromaDB.
    """

    # Combine all retrieved chunks into one context
    context = "\n\n".join(retrieved_documents)

    # Create the final prompt
    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context provided below.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt