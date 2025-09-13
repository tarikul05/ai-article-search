#!/usr/bin/env python3
"""Script to create sample data for testing"""

import os
import sys
import requests
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain.schema import Document
from src.models.search_engine import ArticleSearchEngine, Config

def create_sample_data():
    """Create sample articles for testing"""
    # Load environment variables
    load_dotenv()
    
    # Fetch articles from API endpoint
    api_url = os.getenv("ARTICLE_API_URL", "http://localhost:8000/api/articles")
    try:
        response = requests.get(api_url,verify=False)
        response.raise_for_status()
        res = response.json()
        articles = res.get("data", [])

    except Exception as e:
        print(f"Failed to fetch articles from API: {e}")
        return

    # Convert to LangChain documents
    documents = []
    for article in articles:
        # meta_datas is a list, get first if exists
        meta = article.get("meta_datas", [{}])[0]
        categories = article.get("categories", [])
        tags = article.get("tags", [])
        images = article.get("images", [])
        videos = article.get("videos", [])

        doc = Document(
            page_content=article.get("description", ""),
            metadata={
                "id": article.get("id"),
                "title": article.get("title", ""),
                "author_name": meta.get("author_name", ""),
                "author_organization_name": meta.get("author_organization_name", ""),
                "author_department": meta.get("author_department", ""),
                "categories": ", ".join([cat.get("name", "") for cat in categories]),
                "tags": ", ".join([tag.get("name", "") for tag in tags]),
                "image_url": images[0]["publish_url"] if images else "",
                "video_url": videos[0]["url"] if videos else "",
                "public_status": article.get("public_status", ""),
                "status": article.get("status", ""),
                "priority": article.get("priority", ""),
                "view_count": article.get("view_count", 0),
                "likes_count": article.get("likes_count", 0),
                "comments_count": article.get("comments_count", 0),
                "created_at": article.get("created_at", ""),
                "updated_at": article.get("updated_at", "")
            }
        )
        documents.append(doc)
    
    # Initialize search engine and add documents
    search_engine = ArticleSearchEngine()
    count = search_engine.initialize_database(documents)
    
    print(f"Added {count} document chunks to Chroma database")
    print("Sample data created successfully!")

if __name__ == "__main__":
    create_sample_data()