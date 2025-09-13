"""Configuration management for the application"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Config:
    """Application configuration"""
    # OpenAI settings
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_embedding_model: str = "text-embedding-ada-002"
    
    # Chroma settings
    chroma_persist_dir: str = "./chroma_data"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    
    # Application settings
    embedding_batch_size: int = 100
    max_search_results: int = 10
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        load_dotenv()
        
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "10")),
        )