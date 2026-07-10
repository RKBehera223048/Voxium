from __future__ import annotations

import logging
import asyncio
import random
import math
from typing import List, Any

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """
    Provides text embeddings using a Llama instance via llama-cpp-python,
    or falls back to a simple random/TF-IDF based approach if the model is not
    provided or does not support embeddings.
    """

    def __init__(self, llm_instance: Any = None, dim: int = 384):
        """
        Initializes the EmbeddingProvider.

        Args:
            llm_instance (Any): An instance of Llama (from llama_cpp) to generate embeddings.
                                Defaults to None.
            dim (int): The embedding dimensionality.
        """
        self.llm_instance = llm_instance
        self.vector_dim = dim

    async def embed(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single string.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        if not text:
            logger.warning("Empty text provided for embedding. Returning zero vector.")
            return [0.0] * self.vector_dim

        if self.llm_instance is not None:
            def _create_embedding():
                try:
                    # llama-cpp-python API: create_embedding(text)
                    response = self.llm_instance.create_embedding(text)
                    # response format typical of openai/llama_cpp:
                    # {"data": [{"embedding": [0.1, 0.2, ...]}], ...}
                    if response and 'data' in response and len(response['data']) > 0:
                        return response['data'][0]['embedding']
                except Exception as e:
                    logger.error(f"Error creating embedding with LLM instance: {e}")
                return None

            logger.debug("Requesting embedding from LLM instance.")
            embedding = await asyncio.to_thread(_create_embedding)
            if embedding is not None:
                return embedding
            else:
                logger.warning("Falling back to dummy embedding due to failure in LLM instance.")

        logger.debug("Generating fallback embedding.")
        return self._generate_dummy_embedding(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of strings.

        Args:
            texts (List[str]): The batch of texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(f"Embedding batch of size {len(texts)}.")
        # Process the batch concurrently
        tasks = [self.embed(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        return list(embeddings)

    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """
        Generates a deterministic but pseudo-random dummy embedding.
        Useful for fallback or testing when a model isn't available.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: A normalized dummy vector.
        """
        # Seed the random generator with the text's hash to make it deterministic
        # for a given string, mimicking the behaviour of a real embedding slightly better.
        random.seed(hash(text))
        vector = [random.uniform(-1, 1) for _ in range(self.vector_dim)]
        
        # Normalize the vector
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
            
        # Reset random seed
        random.seed()
        
        return vector
