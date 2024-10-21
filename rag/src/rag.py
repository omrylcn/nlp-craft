import uuid
from datetime import datetime
import logging
import numpy as np
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from collections import deque
from .embedding import EmbeddingFactory
from .llm import LLM
from .db import get_db_connection,init_db,save_conversation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, retriever, llm, db_connection=None,last_question_count=20,embedding_type="SentenceTransformer"):
        self.retriever = retriever
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.db_connection = db_connection
        self.initialize_db()
        self.question_queue = deque([],last_question_count)
        self.embedding_model = EmbeddingFactory().get_embedding_model(embedding_type)

    def initialize_db(self):
        init_db()
        if self.db_connection is None:
            self.db_connection = get_db_connection()

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_chain(self):
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run(self, question):
        try:
            chain = self.create_chain()
            start_time = datetime.now()
            answer = chain.invoke(question)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            self.question_queue.append(self.embedding_model.embed(question))

    
            # Prepare answer data
            answer_data = {
                "answer": answer,
                "model_used": self.llm.model_name,
                "response_time": response_time,
                "relevance": self.evaluate_relevance(question, answer),
                "prompt_tokens": self.llm.get_num_tokens(question),
                "completion_tokens": self.llm.get_num_tokens(answer),
                "total_tokens": self.llm.get_num_tokens(question) + self.llm.get_num_tokens(answer),
                "openai_cost": self.calculate_cost(self.llm.get_num_tokens(question), self.llm.get_num_tokens(answer)),
                "question_trend" : self.analyze_question_trend()
            }

            # Save conversation
            conversation_id = str(uuid.uuid4())
            save_conversation(conversation_id, question, answer_data)

            logger.info(f"RAG process completed for question: {question[:50]}...")
            return answer, answer_data, conversation_id

        except Exception as e:
            logger.error(f"Error in RAG process: {e}")
            raise

    def evaluate_relevance(self, question, answer):
        # Implement relevance evaluation logic here
        # This could be a simple heuristic or a more complex model
        return "RELEVANT"  # Placeholder

    def calculate_cost(self, prompt_tokens, completion_tokens):
        # Implement cost calculation based on your pricing model
        return (prompt_tokens * 0.0001 + completion_tokens * 0.0002) / 1000  # Example calculation

    def get_recent_conversations(self, limit=5):
        # This method could be used to retrieve recent conversations from the database
        from rag.src.db import get_recent_conversations
        return get_recent_conversations(limit)
    
    def analyze_question_trend(self, window_size=5):
        """
        Calculate trend based on the last few question embeddings.
        """
        if len(self.question_queue) < window_size:
            logger.warning("Not enough questions for trend calculation")
            return 0.0 # Return zero vector if not enough data

        recent_embeddings = list(self.question_queue)[-window_size:]
        trend = np.mean(recent_embeddings, axis=0)
        return float(np.mean(trend))