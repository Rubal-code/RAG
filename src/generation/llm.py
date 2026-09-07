import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load .env
load_dotenv()


class LLM:

    def __init__(self):

        # Get Hugging Face token
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise ValueError(
                "HF_TOKEN is not set. Check your .env file."
            )

        # Explicitly use the Featherless AI provider.
        # We are doing this because your live model check
        # showed Qwen/Qwen2.5-7B-Instruct is available there.
        self.client = InferenceClient(
            provider="featherless-ai",
            api_key=hf_token
        )

        # Verified from your live model list
        self.model = "Qwen/Qwen2.5-7B-Instruct"


    def generate(self, prompt):

        # Send the RAG prompt to the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.2
        )

        # Extract the generated answer
        return response.choices[0].message.content