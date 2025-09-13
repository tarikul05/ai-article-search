#!/usr/bin/env python3
"""Test script to verify Chroma and OpenAI setup"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.search_engine import ArticleSearchEngine, Config

def test_setup():
    """Test that Chroma and OpenAI are properly configured"""
    print("Testing setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Test configuration
    try:
        config = Config()
        print("✓ Configuration loaded successfully")
        print(f"  OpenAI model: {config.openai_model}")
        print(f"  Chroma path: {config.chroma_persist_dir}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Test OpenAI connection
    try:
        import openai
        client = openai.OpenAI(api_key=config.openai_api_key)
        models = client.models.list()
        print("✓ OpenAI connection successful")
    except Exception as e:
        print(f"✗ OpenAI connection failed: {e}")
        return False
    
    # Test Chroma connection
    try:
        search_engine = ArticleSearchEngine(config)
        collections = search_engine.list_collections()
        print("✓ Chroma connection successful")
        print(f"  Collections: {collections}")
    except Exception as e:
        print(f"✗ Chroma connection failed: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    return True

if __name__ == "__main__":
    test_setup()