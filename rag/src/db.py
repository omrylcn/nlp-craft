# db.py
import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

tz = ZoneInfo("Europe/Istanbul")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            database=os.getenv("POSTGRES_DB", "course_assistant"),
            user=os.getenv("POSTGRES_USER", "your_username"),
            password=os.getenv("POSTGRES_PASSWORD", "your_password"),
        )
    except psycopg2.Error as e:
        logger.error(f"Unable to connect to the database: {e}")
        raise

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    openai_cost FLOAT,
                    question_trend FLOAT,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        logger.info("Database initialized successfully")
    except psycopg2.Error as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

def save_conversation(conversation_id, question, answer_data):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                prompt_tokens, completion_tokens, total_tokens, openai_cost, question_trend)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                conversation_id,
                question,
                answer_data['answer'],
                answer_data['model_used'],
                answer_data['response_time'],
                answer_data['relevance'],
                answer_data['prompt_tokens'],
                answer_data['completion_tokens'],
                answer_data['total_tokens'],
                answer_data['openai_cost'],
                answer_data["question_trend"]
            ))
        conn.commit()
        logger.info(f"Conversation {conversation_id} saved successfully")
    except psycopg2.Error as e:
        logger.error(f"Error saving conversation: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_recent_conversations(limit=5):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT * FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            return cur.fetchall()
    except psycopg2.Error as e:
        logger.error(f"Error retrieving recent conversations: {e}")
        return []
    finally:
        conn.close()

def drop_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")
        conn.commit()
        logger.info("Database dropped successfully")
    except psycopg2.Error as e:
        logger.error(f"Error dropping database: {e}")
        conn.rollback()
    finally:
        conn.close()