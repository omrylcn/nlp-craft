import os
from abc import ABC, abstractmethod
import logging
from typing import List, Union

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        pass

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str = "text-embedding-ada-002"):
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            if isinstance(text, str):
                text = [text]
            response = self.client.embeddings.create(input=text, model=self.model)
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error in OpenAI embedding: {e}")
            raise

class SentenceTransformerModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name,device=device)
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer with model {model_name}: {e}")
            raise

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            embeddings = self.model.encode(text,batch_size=1,show_progress_bar=False)
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            logger.error(f"Error in SentenceTransformer embedding: {e}")
            raise
        logger.info(f"Registered new embedding model: {name}")

class EmbeddingFactory:
    embedding_models = {
        "OpenAIEmbeddings": OpenAIEmbeddingModel,
        "SentenceTransformer": SentenceTransformerModel
    }

    @classmethod
    def get_embedding_model(cls, embedding_name: str) -> EmbeddingModel:
        if embedding_name not in cls.embedding_models:
            raise ValueError(f"Unsupported embedding model: {embedding_name}")
        
        return cls.embedding_models[embedding_name]()

    @classmethod
    def get_embedding_names(cls) -> list[str]:
        return list(cls.embedding_models.keys())

    @classmethod
    def register_embedding_model(cls, name: str, model_class: type[EmbeddingModel]):
        if not issubclass(model_class, EmbeddingModel):
            raise TypeError(f"{model_class.__name__} must be a subclass of EmbeddingModel")
        cls.embedding_models[name] = model_class
        logger.info(f"Registered new embedding model: {name}")



def get_embedding(text: Union[str, List[str]], embedding_name: str = "OpenAIEmbeddings") -> Union[List[float], List[List[float]]]:
    try:
        embedding_model = EmbeddingFactory.get_embedding_model(embedding_name)
        return embedding_model.embed(text)
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Single sentence embedding
        sentence = "This is a test sentence."
        embedding = get_embedding(sentence)
        logger.info(f"Embedding for single sentence: {embedding[0][:5]}...")  # Show first 5 elements

        # Multiple sentences embedding
        sentences = ["This is the first sentence.", "This is the second sentence."]
        embeddings = get_embedding(sentences)
        logger.info(f"Embeddings for multiple sentences: {[emb[:5] for emb in embeddings]}...")

        # Using a different model
        st_embedding = get_embedding(sentence, embedding_name="SentenceTransformer")
        logger.info(f"SentenceTransformer embedding: {st_embedding[:5]}...")

    except Exception as e:
        logger.error(f"An error occurred: {e}")