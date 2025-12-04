"""
Document Processor for Obligation Tracking System.
Handles document parsing, text extraction, and semantic chunking for various file formats.
"""

import os
import re
import yaml
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import io
import numpy as np

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes various document formats for contract analysis."""
    
    def __init__(self, config_path: str = "config/api_config.yaml"):
        """
        Initialize document processor with configuration.
        
        Args:
            config_path: Path to API configuration YAML file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        doc_config = config.get('document_processing', {})
        self.supported_formats = doc_config.get('supported_formats', ['.pdf', '.docx', '.txt'])
        self.max_file_size_mb = doc_config.get('max_file_size_mb', 50)
        
        # Chunking settings from model config
        with open("config/model_config.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
        
        extraction_settings = model_config.get('extraction_settings', {})
        self.max_chunk_size = extraction_settings.get('chunk_size', 8000)
        self.similarity_threshold = extraction_settings.get('similarity_threshold', 0.75)

        # Initialize embedding model for semantic chunking
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )

        logger.info(f"Document processor initialized with semantic chunking. Supported formats: {self.supported_formats}")
    
    def validate_file(self, filename: str, file_size: int) -> tuple[bool, str]:
        """
        Validate file format and size.
        
        Args:
            filename: Name of the file
            file_size: Size of file in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ext = Path(filename).suffix.lower()
        
        if ext not in self.supported_formats:
            return False, f"Unsupported file format: {ext}. Supported: {self.supported_formats}"
        
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return False, f"File too large: {file_size / (1024*1024):.2f}MB. Max: {self.max_file_size_mb}MB"
        
        return True, ""
    
    def extract_text(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from document based on file type.
        
        Args:
            file_content: Raw file bytes
            filename: Name of the file
            
        Returns:
            Extracted text content
        """
        ext = Path(filename).suffix.lower()
        
        if ext == '.txt':
            return self._extract_from_txt(file_content)
        elif ext == '.pdf':
            return self._extract_from_pdf(file_content)
        elif ext in ['.docx', '.doc']:
            return self._extract_from_docx(file_content)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def _extract_from_txt(self, content: bytes) -> str:
        """Extract text from TXT file."""
        return content.decode('utf-8', errors='ignore')
    
    def _extract_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            logger.warning("pypdf not installed, trying pdfplumber")
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += (page.extract_text() or "") + "\n"
                return text
            except ImportError:
                raise ImportError("Please install pypdf or pdfplumber for PDF support")
    
    def _extract_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("Please install python-docx for DOCX support")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks based on embedding similarity.
        Groups semantically similar sentences together.

        Args:
            text: Full document text

        Returns:
            List of chunk dictionaries with text and metadata
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [{"index": 0, "text": sentences[0], "start_char": 0, "end_char": len(sentences[0])}]

        # Get embeddings for all sentences
        try:
            embeddings = self.embedding_model.embed_documents(sentences)
        except Exception as e:
            logger.error(f"Error getting embeddings for semantic chunking: {e}")
            # Fallback to simple sentence grouping
            return self._fallback_chunk(sentences)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embedding = embeddings[0]
        chunk_index = 0
        chunk_start = 0

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(current_chunk_embedding, embeddings[i])
            current_chunk_text = ' '.join(current_chunk_sentences)

            # Check if we should start a new chunk
            if similarity < self.similarity_threshold or len(current_chunk_text) + len(sentences[i]) > self.max_chunk_size:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append({
                    "index": chunk_index,
                    "text": chunk_text,
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(chunk_text)
                })

                chunk_index += 1
                chunk_start = chunk_start + len(chunk_text) + 1
                current_chunk_sentences = [sentences[i]]
                current_chunk_embedding = embeddings[i]
            else:
                current_chunk_sentences.append(sentences[i])
                # Update embedding as average
                current_chunk_embedding = np.mean([current_chunk_embedding, embeddings[i]], axis=0).tolist()

        # Add last chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                "index": chunk_index,
                "text": chunk_text,
                "start_char": chunk_start,
                "end_char": chunk_start + len(chunk_text)
            })

        logger.info(f"Semantic chunking: Split document into {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks

    def _fallback_chunk(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Fallback chunking when embeddings fail - groups sentences by size."""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        chunk_start = 0

        for sentence in sentences:
            if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "index": chunk_index,
                    "text": chunk_text,
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(chunk_text)
                })
                chunk_index += 1
                chunk_start += len(chunk_text) + 1
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += len(sentence)

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "index": chunk_index,
                "text": chunk_text,
                "start_char": chunk_start,
                "end_char": chunk_start + len(chunk_text)
            })

        return chunks
    
    def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Full document processing pipeline.
        
        Args:
            file_content: Raw file bytes
            filename: Name of the file
            
        Returns:
            Processed document with text and chunks
        """
        is_valid, error = self.validate_file(filename, len(file_content))
        if not is_valid:
            raise ValueError(error)
        
        text = self.extract_text(file_content, filename)
        chunks = self.chunk_text(text)
        
        return {
            "filename": filename,
            "full_text": text,
            "chunks": chunks,
            "total_characters": len(text),
            "total_chunks": len(chunks)
        }

