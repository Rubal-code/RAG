from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Load the embedding model.
        """

        self.model = SentenceTransformer(model_name)


    def embed(self, texts):
        """
        Convert text into embeddings.
        """

        return self.model.encode(
            texts,
            convert_to_numpy=True
        )