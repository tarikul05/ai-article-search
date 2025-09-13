#!/usr/bin/env python3
"""Script to create sample data for testing"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain.schema import Document
from src.models.search_engine import ArticleSearchEngine, Config

def create_sample_data():
    """Create sample articles for testing"""
    # Load environment variables
    load_dotenv()
    
    articles = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            "author": "Dr. Jane Smith",
            "category": "AI"
        },
        {
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, and recurrent neural networks have been applied to fields including computer vision, speech recognition, and natural language processing.",
            "author": "Prof. Alan Turing",
            "category": "AI"
        },
        {
            "title": "Natural Language Processing Trends",
            "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. Recent advances in transformer models like BERT and GPT have revolutionized the field, enabling state-of-the-art performance on various NLP tasks.",
            "author": "Dr. Emily Chen",
            "category": "NLP"
        },
        {
            "title": "Computer Vision Applications",
            "content": "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Applications include facial recognition, object detection, and autonomous vehicles.",
            "author": "Dr. Michael Rodriguez",
            "category": "Computer Vision"
        }
    ]
    
    # Convert to LangChain documents
    documents = []
    for article in articles:
        doc = Document(
            page_content=article["content"],
            metadata={
                "title": article["title"],
                "author": article["author"],
                "category": article["category"]
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