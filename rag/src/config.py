# config.py
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    MODEL_NAME = "gpt-4o-mini"
    CHROMA_COLLECTION_NAME = "rag_collection"
    CHROMA_PORT = 8000
    CHROMA_HOST = "localhost"