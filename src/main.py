#!/usr/bin/env python3
"""
Main application for article search with Chroma and OpenAI
"""

import os
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from src.utils.config import Config
from src.models.search_engine import ArticleSearchEngine

# Initialize rich console
console = Console()

def setup_environment():
    """Load environment variables and validate setup"""
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY not found in environment variables[/red]")
        console.print("Please add your OpenAI API key to the .env file")
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
            console.print("[yellow]Initializing Chroma database...[/yellow]")
            search_engine.initialize_database()
            console.print("[green]Chroma database initialized successfully![/green]")
            
        elif args.list:
            collections = search_engine.list_collections()
            console.print(Panel.fit("Available Collections", style="blue"))
            for collection in collections:
                console.print(f"• {collection}")
                
        elif args.query:
            console.print(f"[blue]Searching for: '{args.query}'[/blue]")
            results = search_engine.search_articles(args.query, top_k=args.top)
            
            if results:
                console.print(Panel.fit(f"Found {len(results)} results", style="green"))
                for i, result in enumerate(results, 1):
                    console.print(f"\n[i]{i}. {result['title']}[/i]")
                    console.print(f"   [dim]Author: {result['author']}[/dim]")
                    console.print(f"   [green]Match: {result['match_percentage']}%[/green]")
                    console.print(f"   [yellow]Preview: {result['content'][:150]}...[/yellow]")
            else:
                console.print("[red]No results found[/red]")
                
        else:
            # Interactive mode
            console.print(Panel.fit("Article Search Engine", style="blue"))
            console.print("Type your search query or 'quit' to exit")
            
            while True:
                try:
                    query = console.input("\n[bold]Search: [/bold]").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if query:
                        with Progress() as progress:
                            task = progress.add_task("[cyan]Searching...", total=1)
                            results = search_engine.search_articles(query, top_k=args.top)
                            progress.update(task, completed=1)
                        
                        if results:
                            console.print(Panel.fit(f"Found {len(results)} results", style="green"))
                            for i, result in enumerate(results, 1):
                                console.print(f"\n[i]{i}. {result['title']}[/i]")
                                console.print(f"   [dim]Author: {result['author']}[/dim]")
                                console.print(f"   [green]Match: {result['match_percentage']}%[/green]")
                                console.print(f"   [yellow]Preview: {result['content'][:150]}...[/yellow]")
                        else:
                            console.print("[red]No results found[/red]")
                            
                except KeyboardInterrupt:
                    console.print("\n[yellow]Exiting...[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    
    except Exception as e:
        console.print(f"[red]Failed to initialize search engine: {e}[/red]")

if __name__ == "__main__":
    main()