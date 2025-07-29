from typing import List, Dict, Any, Optional
import logging
from data_ingestion import DataIngestion
from vector_store import VectorStore
from llm_interface import LLMInterface
from config import Config
import os

logger = logging.getLogger(__name__)

class RAGSystem:
    # Main RAG system that combines all components
    
    def __init__(self, embedding_model: Optional[str] = None, llm_model: Optional[str] = None, vector_store: Optional[VectorStore] = None):
        self.config = Config()
        
        # Determine which models to use (passed-in or from config)
        final_embedding_model = embedding_model if embedding_model else self.config.EMBEDDING_MODEL
        final_llm_model = llm_model if llm_model else self.config.GROQ_MODEL
        
        self.data_ingestion = DataIngestion()
        
        if vector_store is not None:
            self.vector_store = vector_store
            logger.info(f"Using provided vector store with embedding model: {self.vector_store.embedding_model_name}")
        else:
            self.vector_store = VectorStore(embedding_model=final_embedding_model)
            logger.info(f"Created new vector store with embedding model: {final_embedding_model}")
        
        self.llm = LLMInterface(model=final_llm_model)
        
        logger.info(f"RAG System initialized with Embedding: {final_embedding_model}, LLM: {final_llm_model}")
    
    def ingest_data_sources(self, data_sources: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        # Ingest data from multiple sources
        results = {
            'total_documents': 0,
            'successful_sources': 0,
            'failed_sources': 0,
            'details': []
        }
        
        for disaster_type, sources in data_sources.items():
            logger.info(f"Processing data for disaster type: {disaster_type}")
            
            all_documents = []
            
            # Process files
            if 'files' in sources:
                for file_path in sources['files']:
                    try:
                        if not os.path.exists(file_path):
                            logger.warning(f"File not found: {file_path}")
                            continue
                            
                        documents = self.data_ingestion.process_file(file_path)
                        if documents:
                            all_documents.extend(documents)
                            results['details'].append({
                                'source': file_path,
                                'type': 'file',
                                'status': 'success',
                                'documents': len(documents)
                            })
                            logger.info(f"Successfully processed {len(documents)} documents from {file_path}")
                        else:
                            logger.warning(f"No documents extracted from {file_path}")
                            
                    except Exception as e:
                        logger.error(f"Failed to process file {file_path}: {str(e)}")
                        results['details'].append({
                            'source': file_path,
                            'type': 'file',
                            'status': 'failed',
                            'error': str(e)
                        })
                        results['failed_sources'] += 1
                        continue
            
            # Process URLs
            if 'urls' in sources:
                for url in sources['urls']:
                    try:
                        documents = self.data_ingestion.process_url(url)
                        if documents:
                            all_documents.extend(documents)
                            results['details'].append({
                                'source': url,
                                'type': 'url',
                                'status': 'success',
                                'documents': len(documents)
                            })
                            logger.info(f"Successfully processed {len(documents)} documents from {url}")
                        else:
                            logger.warning(f"No documents extracted from {url}")
                            
                    except Exception as e:
                        logger.error(f"Failed to process URL {url}: {str(e)}")
                        results['details'].append({
                            'source': url,
                            'type': 'url',
                            'status': 'failed',
                            'error': str(e)
                        })
                        results['failed_sources'] += 1
                        continue
            
            # Add disaster type metadata to all documents
            for doc in all_documents:
                doc['disaster_type'] = disaster_type
            
            # Add documents to vector store in batches
            if all_documents:
                try:
                    success = self.vector_store.add_documents(all_documents)
                    if success:
                        results['total_documents'] += len(all_documents)
                        results['successful_sources'] += 1
                        logger.info(f"Successfully ingested {len(all_documents)} documents for {disaster_type}")
                    else:
                        results['failed_sources'] += 1
                        logger.error(f"Failed to add documents to vector store for {disaster_type}")
                except Exception as e:
                    logger.error(f"Error adding documents to vector store for {disaster_type}: {str(e)}")
                    results['failed_sources'] += 1
        
        return results
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        # Query the RAG system
        try:
            if top_k is None:
                top_k = self.config.TOP_K_RESULTS
            
            logger.info(f"Processing query: {question}")
            
            # Step 1: Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=top_k)
            
            if not relevant_docs:
                logger.warning("No relevant documents found")
                response = self.llm.generate_response(question, [])
            else:
                # Step 2: Generate response using LLM
                response = self.llm.generate_response(question, relevant_docs)
            
            # Step 3: Prepare comprehensive response
            result = {
                'question': question,
                'answer': response['answer'],
                'model_used': response['model'],
                'retrieved_documents': len(relevant_docs),
                'token_usage': response['tokens'],
                'sources': []
            }
            
            # Add source information
            for doc in relevant_docs:
                source_info = {
                    'source': doc['metadata'].get('source', 'Unknown'),
                    'type': doc['metadata'].get('type', 'Unknown'),
                    'similarity_score': doc['similarity_score'],
                    'snippet': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                }
                
                # Add specific metadata
                if 'page' in doc['metadata']:
                    source_info['page'] = doc['metadata']['page']
                if 'row' in doc['metadata']:
                    source_info['row'] = doc['metadata']['row']
                
                result['sources'].append(source_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'question': question,
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'model_used': self.config.GROQ_MODEL,
                'retrieved_documents': 0,
                'token_usage': {'prompt': 0, 'completion': 0, 'total': 0},
                'sources': []
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        # Get system statistics
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            'vector_store': vector_stats,
            'llm_model': self.llm.model,
            'embedding_model': self.config.EMBEDDING_MODEL,
            'chunk_size': self.config.CHUNK_SIZE,
            'top_k_results': self.config.TOP_K_RESULTS,
            'similarity_threshold': self.config.SIMILARITY_THRESHOLD
        }
    
    def initialize_with_sample_data(self) -> Dict[str, Any]:
        # Initialize the system with sample disaster data
        logger.info("Initializing with sample disaster data")
        
        # Use the predefined data sources from config
        return self.ingest_data_sources(self.config.DATA_SOURCES)
    
    def reset_system(self) -> bool:
        # Reset the entire system
        try:
            success = self.vector_store.reset_collection()
            if success:
                logger.info("System reset successfully")
            return success
        except Exception as e:
            logger.error(f"Error resetting system: {str(e)}")
            return False