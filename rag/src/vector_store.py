from chromadb.config import Settings
from chromadb import HttpClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from .config import Config


class VectorStore:
    def __init__(self,config=None,embedding_model = "text-embedding-ada-002"):
        self.config = config if config else Config()
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        settings = Settings(    
                chroma_server_host=self.config.CHROMA_HOST,
                chroma_server_http_port=int(self.config.CHROMA_PORT),
            )
        self.client= HttpClient(
            host=self.config.CHROMA_HOST,
            port=int(self.config.CHROMA_PORT),
        )
        
        self.vector_store = Chroma(
            collection_name=self.config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            client=self.client
        )

    def add_documents(self, documents:list[Document], collection_name=None):
        #collection = self.create_or_get_collection(collection_name)
        self.vector_store.add_documents(documents)

    def get_retriever(self, search_type="similarity", search_kwargs={"k": 6}):
        return self.vector_store.as_retriever(search_type=search_type,search_kwargs=search_kwargs)

    def list_collections(self):
        return self.client.list_collections()