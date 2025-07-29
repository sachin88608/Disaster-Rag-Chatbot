#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
from datetime import datetime
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset
import numpy as np
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevancy,
    ContextRecall,
    ContextPrecision
)
from ragas import evaluate
from dotenv import load_dotenv
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from llm_interface import LLMInterface

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ragas_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GROQLLM:
    # LLM wrapper for GROQ API using LLMInterface (llama-3.1-8b-instant or similar)
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.llm = LLMInterface(model=model_name)
        self.batch_size = 1  # API is stateless, so batch one by one

    def batch_generate(self, prompts: list) -> list:
        responses = []
        for i, prompt in enumerate(prompts):
            try:
                # GROQ API expects a question and context, so we treat the prompt as the user message
                logger.info(f"[GROQ] Processing prompt {i+1}/{len(prompts)} on GROQ API ({self.model_name})")
                result = self.llm.generate_response(prompt, context_docs=[])  # context is in prompt
                answer = result.get('answer', 'Unable to determine.')
                responses.append(answer)
            except Exception as e:
                logger.error(f"[GROQ] Error processing prompt {i+1}: {str(e)}")
                responses.append("Unable to determine.")
        return responses

class RAGASEvaluator:
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        # Initialize the RAGAS evaluator with a GROQ API model
        self.model_name = model_name
        self.llm = GROQLLM(model_name=model_name)
        logger.info(f"Initialized RAGAS evaluator with GROQ model: {model_name}")

    def prepare_dataset(self, evaluation_data: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for evaluation"""
        try:
            # Convert to RAGAS format
            ragas_data = []
            for item in evaluation_data:
                ragas_item = {
                    'question': item['question'],
                    'answer': item['answer'],
                    'contexts': item['contexts'],
                    'ground_truth': item.get('ground_truth', '')
                }
                ragas_data.append(ragas_item)
            
            # Create dataset
            dataset = Dataset.from_list(ragas_data)
            logger.info(f"Created dataset with {len(ragas_data)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise

    def _score_response(self, response: str) -> float:
        # Score a response based on evaluation keywords
        try:
            response_lower = response.lower().strip()
            
            if 'yes' in response_lower and 'no' not in response_lower:
                return 1.0
            elif 'no' in response_lower and 'yes' not in response_lower:
                return 0.0
            elif 'partially' in response_lower or 'somewhat' in response_lower:
                return 0.5
            elif 'unable' in response_lower or 'cannot' in response_lower:
                return 0.0
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error scoring response: {str(e)}")
            return 0.5

    def calculate_metrics_batch(self, dataset: Dataset) -> Dict[str, float]:
        # Calculate all metrics using batch processing
        try:
            logger.info("Starting batch metric calculation...")
            start_time = time.time()
            
            results = {}
            
            # Prepare all prompts for batch processing
            faithfulness_prompts = []
            answer_relevancy_prompts = []
            context_relevancy_prompts = []
            context_recall_prompts = []
            context_precision_prompts = []
            
            for i in range(len(dataset)):
                question = dataset[i]['question']
                answer = dataset[i]['answer']
                contexts = dataset[i]['contexts']
                ground_truth = dataset[i].get('ground_truth', '')
                context_text = ' '.join(contexts)
                
                # Faithfulness prompts
                faithfulness_prompts.append(
                    f"Question: {question}\nAnswer: {answer}\nContext: {context_text}\n\n"
                    "Is the answer supported by the context? Answer ONLY with 'Yes', 'No', or 'Partially'."
                )
                
                # Answer relevancy prompts
                answer_relevancy_prompts.append(
                    f"Question: {question}\nAnswer: {answer}\n\n"
                    "Is the answer relevant to the question? Answer ONLY with 'Yes', 'No', or 'Partially'."
                )
                
                # Context relevancy prompts
                context_relevancy_prompts.append(
                    f"Question: {question}\nContext: {context_text}\n\n"
                    "Is the context relevant to the question? Answer ONLY with 'Yes', 'No', or 'Partially'."
                )
                
                # Context recall prompts
                context_recall_prompts.append(
                    f"Question: {question}\nContext: {context_text}\nGround Truth: {ground_truth}\n\n"
                    "Does the context contain information to answer the question? Answer ONLY with 'Yes', 'No', or 'Partially'."
                )
                
                # Context precision prompts
                context_precision_prompts.append(
                    f"Question: {question}\nContext: {context_text}\n\n"
                    "Is the context precise and focused on the question? Answer ONLY with 'Yes', 'No', or 'Partially'."
                )
            
            # Process all metrics in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all metric calculations
                future_to_metric = {
                    executor.submit(self._calculate_single_metric_batch, faithfulness_prompts, "faithfulness"): "faithfulness",
                    executor.submit(self._calculate_single_metric_batch, answer_relevancy_prompts, "answer_relevancy"): "answer_relevancy",
                    executor.submit(self._calculate_single_metric_batch, context_relevancy_prompts, "context_relevancy"): "context_relevancy",
                    executor.submit(self._calculate_single_metric_batch, context_recall_prompts, "context_recall"): "context_recall",
                    executor.submit(self._calculate_single_metric_batch, context_precision_prompts, "context_precision"): "context_precision"
                }
                
                # Collect results
                for future in as_completed(future_to_metric):
                    metric_name = future_to_metric[future]
                    try:
                        score = future.result()
                        results[metric_name] = score
                        logger.info(f"Completed {metric_name}: {score:.3f}")
                    except Exception as e:
                        logger.error(f"Error calculating {metric_name}: {str(e)}")
                        results[metric_name] = 0.0
            
            # Calculate average score
            valid_scores = [score for score in results.values() if score > 0]
            results['average_score'] = np.mean(valid_scores) if valid_scores else 0.0

            # Calculate weighted score
            results['weighted_score'] = (
                0.30 * results.get('faithfulness', 0.0) +
                0.25 * results.get('answer_relevancy', 0.0) +
                0.20 * results.get('context_relevancy', 0.0) +
                0.15 * results.get('context_recall', 0.0) +
                0.10 * results.get('context_precision', 0.0)
            )
            
            end_time = time.time()
            logger.info(f"Batch metric calculation completed in {end_time - start_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch metric calculation: {str(e)}")
            return {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_relevancy': 0.0,
                'context_recall': 0.0,
                'context_precision': 0.0,
                'average_score': 0.0
            }

    def _calculate_single_metric_batch(self, prompts: List[str], metric_name: str) -> float:
        # Calculate a single metric using batch processing
        try:
            logger.info(f"Calculating {metric_name}...")
            
            # Generate responses in batch
            responses = self.llm.batch_generate(prompts)
            
            # Score responses
            scores = [self._score_response(response) for response in responses]
            
            # Log some examples for debugging
            for i, (prompt, response, score) in enumerate(zip(prompts[:2], responses[:2], scores[:2])):
                logger.info(f"[{metric_name}] Example {i+1}:")
                logger.info(f"[{metric_name}] Prompt: {prompt[:100]}...")
                logger.info(f"[{metric_name}] Response: {response}")
                logger.info(f"[{metric_name}] Score: {score}")
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error calculating {metric_name}: {str(e)}")
            return 0.0

    def evaluate(self, evaluation_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            if evaluation_data is None:
                # Load from CSV file if available
                csv_file = "SAMPLE_EVALUATION_DATASET.csv"
                if os.path.exists(csv_file):
                    logger.info(f"Loading evaluation data from {csv_file}")
                    df = pd.read_csv(csv_file)
                    evaluation_data = []
                    for _, row in df.iterrows():
                        evaluation_data.append({
                            'question': row['question'],
                            'answer': row['answer'],
                            'contexts': [row['context']] if pd.notna(row['context']) else [],
                            'ground_truth': row['ground_truth'] if pd.notna(row['ground_truth']) else ''
                        })
                else:
                    logger.warning(f"CSV file {csv_file} not found, using sample data")
                    evaluation_data = []
            
            logger.info("Starting OPTIMIZED RAGAS evaluation...")
            print(f"\nOPTIMIZED EVALUATION")
            print("=" * 50)
            print(f"Number of examples: {len(evaluation_data)}")
            print(f"Model: {self.model_name}")
            print(f"Batch size: {self.llm.batch_size}")
            print("=" * 50)
            
            # Prepare dataset
            dataset = self.prepare_dataset(evaluation_data)
            
            # Calculate metrics using batch processing
            start_time = time.time()
            metrics = self.calculate_metrics_batch(dataset)
            end_time = time.time()
            
            # Prepare results
            results = {
                'metrics': metrics,
                'evaluation_data': evaluation_data,
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'dataset_size': len(evaluation_data),
                'processing_time': end_time - start_time,
                'optimization': 'batch_processing'
            }
            
            # Save results
            self.save_results(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Error in optimized evaluation: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return {
                'metrics': {
                    'faithfulness': 0.0,
                    'answer_relevancy': 0.0,
                    'context_relevancy': 0.0,
                    'context_recall': 0.0,
                    'context_precision': 0.0,
                    'average_score': 0.0
                },
                'evaluation_data': evaluation_data or [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        try:
            # Create results directory
            results_dir = "evaluation_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(results_dir, f"ragas_evaluation_{timestamp}.json")
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("OPTIMIZED RAGAS EVALUATION RESULTS")
            print("="*60)
            print(f"Model: {results.get('model_name', 'Unknown')}")
            print(f"Timestamp: {results['timestamp']}")
            print(f"Dataset size: {results.get('dataset_size', 0)}")
            print(f"Processing time: {results.get('processing_time', 0):.2f} seconds")
            print(f"Optimization: {results.get('optimization', 'Unknown')}")
            print("\nMetric Scores:")
            print("-" * 30)
            
            for metric_name, score in results['metrics'].items():
                if metric_name != 'average_score':
                    print(f"{metric_name.replace('_', ' ').title()}: {score:.3f}")
            
            print(f"\nAverage Score: {results['metrics'].get('average_score', 0):.3f}")
            print(f"Weighted Score: {results['metrics'].get('weighted_score', 0):.3f}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def evaluate_dataset(self, rag_system=None) -> Dict[str, Any]:
        # Evaluate a dataset using the provided RAG system
        try:
            # Load evaluation dataset - try new dataset first, fallback to old one
            csv_files = ["VECTORSTORE_EVALUATION_DATASET.csv", "SAMPLE_EVALUATION_DATASET.csv"]
            csv_file = None
            
            for file in csv_files:
                if os.path.exists(file):
                    csv_file = file
                    break
            
            if csv_file is None:
                raise FileNotFoundError(f"Evaluation dataset not found. Tried: {csv_files}")
            
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded evaluation dataset from {csv_file} with {len(df)} questions")
            
            # Prepare evaluation data
            evaluation_data = []
            for _, row in df.iterrows():
                question = row['question']
                ground_truth = row.get('ground_truth_answer', '')
                
                # Get answer from RAG system
                try:
                    response = rag_system.query(question)
                    chatbot_answer = response['answer']
                    retrieved_contexts = [source['snippet'] for source in response['sources']]
                    
                    # Truncate long contexts
                    max_context_length = 500
                    truncated_contexts = []
                    for ctx in retrieved_contexts:
                        if len(ctx) > max_context_length:
                            truncated_contexts.append(ctx[:max_context_length] + "...")
                        else:
                            truncated_contexts.append(ctx)
                    
                    evaluation_data.append({
                        'question': question,
                        'answer': chatbot_answer,
                        'contexts': truncated_contexts,
                        'ground_truth': ground_truth
                    })
                    
                except Exception as e:
                    logger.warning(f"Error getting answer for question: {e}")
                    # Use ground truth as fallback
                    evaluation_data.append({
                        'question': question,
                        'answer': ground_truth,
                        'contexts': [str(ground_truth)],
                        'ground_truth': ground_truth
                    })
            
            # Run evaluation
            results = self.evaluate(evaluation_data)
            
            # Add metadata
            results['timestamp'] = datetime.now().isoformat()
            results['model_name'] = self.model_name
            results['dataset_size'] = len(evaluation_data)
            results['dataset_source'] = csv_file
            
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluate_dataset: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    # Main function for testing optimized evaluation
    print("OPTIMIZED RAGAS EVALUATION TEST")
    print("=" * 60)
    
    # Test the evaluator with a model that works well with simple prompts
    evaluator = RAGASEvaluator(model_name="llama-3.1-8b-instant")
    
    # Run evaluation
    results = evaluator.evaluate()
    
    print(f"\n OPTIMIZED EVALUATION COMPLETED!")
    print(f"Total time: {results.get('processing_time', 0):.2f} seconds")
    print(f"Average score: {results['metrics'].get('average_score', 0):.3f}")

if __name__ == "__main__":
    main() 