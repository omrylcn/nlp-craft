from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from .config import Config


class LLM:
    def __init__(self,config=None):
        self.config = config if config else Config()
        self.model = ChatOpenAI(model=self.config.MODEL_NAME)


    
