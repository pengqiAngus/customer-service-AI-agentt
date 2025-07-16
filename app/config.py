import os
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

class Config(BaseSettings):
    """Application configuration from environment variables"""
    
    # Flask settings
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 5000))
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Model training settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
    TRAINING_DATA_PATH: str = os.getenv("TRAINING_DATA_PATH", "./data/training_dataset.json")
    OUTPUT_MODEL_PATH: str = os.getenv("OUTPUT_MODEL_PATH", "./models/fine_tuned_model")
    LORA_RANK: int = int(os.getenv("LORA_RANK", 16))
    LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", 32))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 5e-5))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", 3))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 4))
    
    # RAG settings
    KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_files/")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store/")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_NAME: str = os.getenv("DB_NAME", "customer_service")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Text2SQL settings
    ENABLE_TEXT2SQL: bool = os.getenv("ENABLE_TEXT2SQL", "True").lower() == "true"
    SQL_TABLES_INFO_PATH: str = os.getenv("SQL_TABLES_INFO_PATH", "./data/database_schema.json")
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    MEMORY_TTL: int = int(os.getenv("MEMORY_TTL", 3600))
    
    # Agent settings
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", 10))
    AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", 0.7))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 1000))

config = Config()