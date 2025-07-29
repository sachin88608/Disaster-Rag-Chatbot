import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from typing import List, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    # Handles ingestion of various data formats
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.csv', '.xlsx', '.xls', '.txt', '.md']
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        # Process PDF files and extract text
        try:
            documents = []
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append({
                        'content': text,
                        'source': file_path,
                        'page': page_num + 1,
                        'type': 'pdf'
                    })
            
            logger.info(f"Processed PDF: {file_path}, Pages: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        # Process CSV files
        try:
            df = pd.read_csv(file_path)
            documents = []
            
            # Convert each row to a document
            for idx, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                documents.append({
                    'content': content,
                    'source': file_path,
                    'row': idx + 1,
                    'type': 'csv'
                })
            
            logger.info(f"Processed CSV: {file_path}, Rows: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}")
            return []
    
    def process_excel(self, file_path: str) -> List[Dict[str, Any]]:
        # Process Excel files
        try:
            documents = []
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                for idx, row in df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    documents.append({
                        'content': content,
                        'source': file_path,
                        'sheet': sheet_name,
                        'row': idx + 1,
                        'type': 'excel'
                    })
            
            logger.info(f"Processed Excel: {file_path}, Documents: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {str(e)}")
            return []
    
    def process_text(self, file_path: str) -> List[Dict[str, Any]]:
        # Process text files
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if content.strip():
                return [{
                    'content': content,
                    'source': file_path,
                    'type': 'text'
                }]
            return []
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def process_url(self, url: str) -> List[Dict[str, Any]]:
        # Process web URLs
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Try to detect the encoding
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding
            
            # Parse with explicit encoding
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text with better encoding handling
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if text.strip():
                return [{
                    'content': text,
                    'source': url,
                    'type': 'url'
                }]
            return []
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return []
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        # Process a single file based on its extension
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.process_pdf(file_path)
        elif file_ext == '.csv':
            return self.process_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return self.process_excel(file_path)
        elif file_ext in ['.txt', '.md']:
            return self.process_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        # Process all supported files in a directory
        all_documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                documents = self.process_file(str(file_path))
                all_documents.extend(documents)
        
        logger.info(f"Processed directory: {directory_path}, Total documents: {len(all_documents)}")
        return all_documents