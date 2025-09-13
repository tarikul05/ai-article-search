#!/usr/bin/env python3
"""
Main application for article search with Chroma and OpenAI
"""

import os
import argparse
from dotenv import load_dotenv

from src.models.search_engine import ArticleSearchEngine, Config

def setup_environment():
    """Load environment variables and validate setup"""
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file")
        return False
    
    return True

def main():
    """Main application entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Article Search with Chroma and OpenAI")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--init", "-i", action="store_true", help="Initialize Chroma database")
    parser.add_argument("--list", "-l", action="store_true", help="List available collections")
    parser.add_argument("--top", "-t", type=int, default=5, help="Number of results to show")
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        return
    
    # Initialize search engine
    try:
        search_engine = ArticleSearchEngine()
        
        if args.init:
            print("Initializing Chroma database...")
            search_engine.initialize_database()
            print("Chroma database initialized successfully!")
            
        elif args.list:
            collections = search_engine.list_collections()
            print("Available Collections:")
            for collection in collections:
                print(f"• {collection}")
                
        elif args.query:
            print(f"Searching for: '{args.query}'")
            results = search_engine.search_articles(args.query, top_k=args.top)
            
            if results:
                print(f"Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['title']}")
                    print(f"   Author: {result['author']}")
                    print(f"   Match: {result['match_percentage']}%")
                    print(f"   Preview: {result['content'][:150]}...")
            else:
                print("No results found")
                
        else:
            # Interactive mode
            print("Article Search Engine")
            print("Type your search query or 'quit' to exit")
            
            while True:
                try:
                    query = input("\nSearch: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if query:
                        results = search_engine.search_articles(query, top_k=args.top)
                        
                        if results:
                            print(f"Found {len(results)} results")
                            for i, result in enumerate(results, 1):
                                print(f"\n{i}. {result['title']}")
                                print(f"   Author: {result['author']}")
                                print(f"   Match: {result['match_percentage']}%")
                                print(f"   Preview: {result['content'][:150]}...")
                        else:
                            print("No results found")
                            
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
    except Exception as e:
        print(f"Failed to initialize search engine: {e}")

if __name__ == "__main__":
    main()