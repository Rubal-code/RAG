import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load .env
load_dotenv()


class LLM:

    def __init__(
        self,
        model,
        provider,
        max_tokens=300,
        temperature=0.2
    ):
        """
        Create the Hugging Face LLM client.
        """

        # Get HF token from .env
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise ValueError(
                "HF_TOKEN is not set. Check your .env file."
            )

        # Hugging Face client
        self.client = InferenceClient(
            provider=provider,
            api_key=hf_token
        )

        # Generation settings
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature


    def generate(self, prompt):
        """
        Generate an answer using the LLM.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content