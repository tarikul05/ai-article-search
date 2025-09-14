"""Article search engine using Chroma and OpenAI"""

import os
import chromadb
import time
from typing import List, Dict, Optional
from chromadb.config import Settings
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Config:
    """Application configuration"""
    def __init__(self):
        # OpenAI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # Chroma settings
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        # Application settings
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "10"))


class ArticleSearchEngine:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.client = None
        self.embeddings = None
        self.collection = None
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize OpenAI and Chroma clients"""
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.openai_embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=self.config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("articles")
        except:
            self.collection = self.client.create_collection("articles")
    
    def initialize_database(self, documents: Optional[List[Document]] = None):
        """Initialize the database with sample data or provided documents"""
        if documents is None:
            documents = self._import_articles_via_api()
            # if not documents:
            #     documents = self._create_sample_documents()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add documents to Chroma
        ids = [f"doc_{i}" for i in range(len(chunks))]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings and add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return len(chunks)

    def _import_articles_via_api(self):
        """Import articles from an API endpoint"""
        import requests
        api_url = os.getenv("ARTICLE_API_URL", "http://localhost:8000/api/articles")
        print(f"Fetching articles from API: {api_url}")
        
        try:
            response = requests.get(api_url, verify=False)
            response.raise_for_status()
            res = response.json()
            articles = res.get("data", [])
            print(f"Fetched {len(articles)} articles from API")
        except Exception as e:
            print(f"Failed to fetch articles from API: {e}")
            return 0
        
        # Convert to LangChain documents
        documents = []
        for article in articles:
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
        return documents

    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents for testing"""
        return [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                metadata={
                    "title": "Introduction to Machine Learning",
                    "author": "John Doe",
                    "category": "AI"
                }
            ),
            Document(
                page_content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                metadata={
                    "title": "Deep Learning Fundamentals",
                    "author": "Jane Smith",
                    "category": "AI"
                }
            ),
            Document(
                page_content="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
                metadata={
                    "title": "NLP Overview",
                    "author": "Alan Johnson",
                    "category": "NLP"
                }
            )
        ]
    
    def search_articles(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for articles similar to the query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Process results
        articles = []
        for i in range(len(results['ids'][0])):
            article_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            document = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            
            # Calculate match percentage
            match_percentage = self._calculate_match_percentage(distance)
            
            articles.append({
                'id': article_id,
                'title': metadata.get('title', 'Untitled'),
                'content': document,
                'author_name': metadata.get('author_name', 'Unknown'),
                'categories': metadata.get('categories', 'Unknown'),
                'tags': metadata.get('tags', 'Unknown'),
                'match_percentage': match_percentage,
                'distance': distance
            })
        
        return articles
    
    def _calculate_match_percentage(self, distance: float) -> float:
        """Convert distance to match percentage (0-100%)"""
        # Chroma uses cosine distance where 0 is perfect match, 2 is completely different
        # Convert to similarity score (0-1) where 1 is perfect match
        similarity = 1 - (distance / 2)
        
        # Convert to percentage and ensure it's within bounds
        match_percentage = max(0, min(100, similarity * 100))
        return round(match_percentage, 2)
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        return [col.name for col in self.client.list_collections()]
    
    def add_document(self, title: str, content: str, author: str, category: str = "General"):
        """Add a single document to the collection"""
        document = Document(
            page_content=content,
            metadata={
                "title": title,
                "author": author,
                "category": category
            }
        )
        
        # Generate ID based on timestamp
        doc_id = f"doc_{int(time.time() * 1000)}"
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            documents=[document.page_content],
            metadatas=[document.metadata]
        )
        
        return doc_id