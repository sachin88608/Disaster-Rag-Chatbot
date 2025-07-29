#!/usr/bin/env python3
"""
Setup script to load data once with the default embedding model.
This creates the base vector store that will be used for all experiments.
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import VectorStore
from rag_system import RAGSystem
from config import Config

def setup_data():
    # Load data once with the default embedding model
    try:
        print("Setting up data for experiments...")
        print("=" * 60)
        
        # Get default embedding model from config
        config = Config()
        default_embedding_model = config.EMBEDDING_MODEL
        
        print(f" Using default embedding model: {default_embedding_model}")
        print(f" Data will be stored in: {config.CHROMA_PERSIST_DIR}")
        print(f" Collection name: {config.COLLECTION_NAME}")
        print()
        
        # Initialize vector store with default model
        print(" Initializing vector store...")
        vector_store = VectorStore(embedding_model=default_embedding_model)
        
        # Check if data already exists
        stats = vector_store.get_collection_stats()
        total_docs = stats.get('total_documents', 0)
        
        if total_docs > 0:
            print(f" Vector store already has {total_docs} documents")
            print(" Data is ready for experiments!")
            return True
        
        # Load data if not already present
        print(" Loading data into vector store...")
        rag = RAGSystem(embedding_model=default_embedding_model, vector_store=vector_store)
        
        # Load sample data
        result = rag.initialize_with_sample_data()
        
        if result and result.get('total_documents', 0) > 0:
            print(f" Successfully loaded {result['total_documents']} documents")
            print(" Data is ready for experiments!")
            return True
        else:
            print(" Failed to load data")
            return False
            
    except Exception as e:
        print(f" Error setting up data: {str(e)}")
        return False

def main():
    print("Disaster Management Data Setup---")
    print("=" * 60)
    
    success = setup_data()
    
    if success:
        print("\n Data setup completed successfully!")
        print("You can now run experiments with different embedding models.")
        print("The vector store will automatically re-embed documents for each experiment.")
    else:
        print("\n Data setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()