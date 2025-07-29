"""
Script to run a single RAGAS experiment by experiment number.
This allows running experiments one by one instead of all at once.
Uses OPTIMIZED RAGAS evaluation for better performance.
DYNAMIC WORKFLOW: Re-embeds all documents with the experiment's embedding model.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
from ragas_evaluation_fixed import RAGASEvaluator  # Using optimized version
from config import Config
from vector_store import VectorStore
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('single_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(experiment_number: int):
    # Run a single experiment by its number (1-16) with dynamic re-embedding
    try:
        if not 1 <= experiment_number <= 16:
            raise ValueError("Experiment number must be between 1 and 16")
        
        print(f"Starting Experiment {experiment_number} (DYNAMIC RE-EMBEDDING)")
        print("="*60)
        
        # Define experiment configurations
        experiments = [
            {
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'llama-3.3-70b-versatile',
                'description': 'all-MiniLM-L6-v2 + llama-3.3-70b-versatile'
            },
            {
                'embedding_model': 'all-MiniLM-L12-v2',
                'llm_model': 'llama-3.3-70b-versatile',
                'description': 'all-MiniLM-L12-v2 + llama-3.3-70b-versatile'
            },
            {
                'embedding_model': 'BAAI/bge-small-en-v1.5',
                'llm_model': 'llama-3.3-70b-versatile',
                'description': 'BAAI/bge-small-en-v1.5 + llama-3.3-70b-versatile'
            },
            {
                'embedding_model': 'thenlper/gte-small',
                'llm_model': 'llama-3.3-70b-versatile',
                'description': 'thenlper/gte-small + llama-3.3-70b-versatile'
            },
            {
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'llama-3.1-8b-instant',
                'description': 'all-MiniLM-L6-v2 + llama-3.1-8b-instant'
            },
            {
                'embedding_model': 'all-MiniLM-L12-v2',
                'llm_model': 'llama-3.1-8b-instant',
                'description': 'all-MiniLM-L12-v2 + llama-3.1-8b-instant'
            },
            {
                'embedding_model': 'BAAI/bge-small-en-v1.5',
                'llm_model': 'llama-3.1-8b-instant',
                'description': 'BAAI/bge-small-en-v1.5 + llama-3.1-8b-instant'
            },
            {
                'embedding_model': 'thenlper/gte-small',
                'llm_model': 'llama-3.1-8b-instant',
                'description': 'thenlper/gte-small + llama-3.1-8b-instant'
            },
            {
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'mistral-saba-24b',
                'description': 'all-MiniLM-L6-v2 + mistral-saba-24b'
            },
            {
                'embedding_model': 'all-MiniLM-L12-v2',
                'llm_model': 'mistral-saba-24b',
                'description': 'all-MiniLM-L12-v2 + mistral-saba-24b'
            },
            {
                'embedding_model': 'BAAI/bge-small-en-v1.5',
                'llm_model': 'mistral-saba-24b',
                'description': 'BAAI/bge-small-en-v1.5 + mistral-saba-24b'
            },
            {
                'embedding_model': 'thenlper/gte-small',
                'llm_model': 'mistral-saba-24b',
                'description': 'thenlper/gte-small + mistral-saba-24b'
            }
        ]
        
        # Get experiment config
        experiment_config = experiments[experiment_number - 1]
        
        print(f"Experiment Configuration:")
        print(f"Embedding Model: {experiment_config['embedding_model']}")
        print(f"LLM Model: {experiment_config['llm_model']}")
        print(f"RAGAS Model: microsoft/DialoGPT-medium (Optimized)")
        print()
        
        # Step 1: Initialize vector store with the experiment's embedding model
        print("Step 1: Initializing vector store with experiment's embedding model...")
        vector_store = VectorStore(embedding_model=experiment_config['embedding_model'])
        
        # Step 2: Check if vector store has data
        stats = vector_store.get_collection_stats()
        total_docs = stats.get('total_documents', 0)
        
        if total_docs == 0:
            print("No data found in vector store!")
            print("Please run 'python setup_data.py' first to load the base data.")
            return None
        
        print(f"Vector store has {total_docs} documents")
        
        # Step 3: Re-embed all documents with the new embedding model
        print(f"Step 2: Re-embedding {total_docs} documents with {experiment_config['embedding_model']}...")
        try:
            vector_store.update_embedding_model(experiment_config['embedding_model'])
            print(f"Successfully re-embedded all documents with {experiment_config['embedding_model']}")
        except Exception as e:
            print(f"Error re-embedding documents: {str(e)}")
            return None
        
        # Step 4: Initialize RAG system with the updated vector store
        print("Step 3: Initializing RAG system...")
        rag = RAGSystem(
            embedding_model=experiment_config['embedding_model'],
            llm_model=experiment_config['llm_model'],
            vector_store=vector_store  # Use the updated vector store
        )
        
        # Step 5: Run RAGAS evaluation
        print("Step 4: Running RAGAS evaluation...")
        evaluator = RAGASEvaluator(model_name="llama-3.1-8b-instant")
        evaluation_result = evaluator.evaluate_dataset(rag)
        
        # Add experiment metadata
        evaluation_result['experiment_number'] = experiment_number
        evaluation_result['configuration'] = experiment_config
        evaluation_result['embedding_model'] = experiment_config['embedding_model']
        evaluation_result['llm_model'] = experiment_config['llm_model']
        evaluation_result['total_documents'] = total_docs
        
        if evaluation_result.get('error'):
            print(f"\nExperiment {experiment_number} failed")
            print(f"Error: {evaluation_result['error']}")
            return None
        
        # Save results
        save_experiment_results(evaluation_result, experiment_number)
        
        print(f"\n Successfully completed Experiment {experiment_number}!")
        print("\nResults Summary:")
        print("-" * 50)
        print(f"Embedding Model: {evaluation_result.get('embedding_model', 'Unknown')}")
        print(f"LLM Model: {evaluation_result.get('llm_model', 'Unknown')}")
        print(f"Total Documents: {evaluation_result.get('total_documents', 0)}")
        print(f"Processing Time: {evaluation_result.get('processing_time', 0):.2f} seconds")
        print("\nMetric Scores:")
        for metric_name, score in evaluation_result.get('metrics', {}).items():
            if metric_name != 'average_score':
                print(f"  {metric_name.replace('_', ' ').title()}: {score:.3f}")
        print(f"  Average Score: {evaluation_result.get('metrics', {}).get('average_score', 0):.3f}")
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Error running experiment {experiment_number}: {str(e)}")
        print(f"Error: {str(e)}")
        return None

def save_experiment_results(results: dict, experiment_number: int):
    # Save individual experiment results
    try:
        # Create results directory
        results_dir = "batch_experiment_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Experiment_{experiment_number}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Saved experiment results to {filepath}")
        
    except Exception as e:
        print(f"Error saving experiment results: {str(e)}")

def check_data_ready():
    # Check if data is ready for experiments
    try:
        print("Checking if data is ready for experiments...")
        
        # Initialize vector store with default model
        config = Config()
        vector_store = VectorStore(embedding_model=config.EMBEDDING_MODEL)
        
        # Check if data exists
        stats = vector_store.get_collection_stats()
        total_docs = stats.get('total_documents', 0)
        
        if total_docs > 0:
            print(f"Data is ready! Vector store has {total_docs} documents")
            return True
        else:
            print("No data found in vector store!")
            print("Please run 'python setup_data.py' first to load the base data.")
            return False
            
    except Exception as e:
        print(f"Error checking data: {str(e)}")
        return False

def main():
    # Main execution function
    try:
        print("Dynamic RAGAS Experiment Runner")
        print("=" * 60)
        print("This script runs experiments with dynamic re-embedding.")
        print("Each experiment re-embeds all documents with a new embedding model.")
        print("=" * 60)
        
        # Check if data is ready
        if not check_data_ready():
            return
        
        # Get experiment number from user
        while True:
            try:
                experiment_number = int(input("\nEnter experiment number (1-16): "))
                if 1 <= experiment_number <= 16:
                    break
                print("Please enter a number between 1 and 16")
            except ValueError:
                print("Please enter a valid number")
        
        # Run the experiment
        result = run_experiment(experiment_number)
        
        if result:
            print(f"\n Experiment {experiment_number} completed successfully!")
            print(f" Average Score: {result.get('metrics', {}).get('average_score', 0):.3f}")
        else:
            print(f"\n Experiment {experiment_number} failed")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 