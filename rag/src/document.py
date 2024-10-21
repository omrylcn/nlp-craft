import bs4
from src.config import Config
from langchain_community.document_loaders import WebBaseLoader

class DocumentLoader:
    @staticmethod
    def load_web_documents(web_paths):
        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        return loader.load()
    
# document_processor.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=2000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            add_start_index=True
        )

    def split_documents(self, docs):
        return self.text_splitter.split_documents(docs)
    

