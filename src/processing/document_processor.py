"""
Document Processor for Obligation Tracking System.
Handles document parsing, text extraction, and semantic chunking for various file formats.
Optimized for contract documents with section-aware chunking.
"""

import os
import re
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
import numpy as np

logger = logging.getLogger(__name__)

# Contract section patterns for intelligent chunking
CONTRACT_SECTION_PATTERNS = [
    r'(?i)^(?:ARTICLE|SECTION|CLAUSE|PART)\s+[\dIVXivx]+[.\s:]',  # ARTICLE 1, Section 2.1
    r'(?i)^\d+\.\d+(?:\.\d+)?\s+[A-Z]',  # 1.1 Title, 2.3.1 Subsection
    r'(?i)^(?:WHEREAS|NOW,?\s*THEREFORE|IN WITNESS WHEREOF)',  # Legal phrases
    r'(?i)^(?:DEFINITIONS|TERM|PAYMENT|TERMINATION|CONFIDENTIALITY|INDEMNIFICATION|WARRANTIES)',  # Common sections
    r'(?i)^(?:SCHEDULE|EXHIBIT|APPENDIX|ANNEX)\s+[A-Z\d]',  # Attachments
]


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
        self.chunk_overlap = extraction_settings.get('chunk_overlap', 500)
        self.similarity_threshold = extraction_settings.get('similarity_threshold', 0.75)

        # Initialize embedding model for semantic chunking
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )

        logger.info(f"Document processor initialized with contract-aware chunking. Max chunk: {self.max_chunk_size}, Overlap: {self.chunk_overlap}")
    
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

    def _is_section_boundary(self, text: str) -> bool:
        """Check if text starts with a contract section header."""
        text = text.strip()
        for pattern in CONTRACT_SECTION_PATTERNS:
            if re.match(pattern, text):
                return True
        return False

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split contract text into logical sections based on headers.
        Returns list of (section_header, section_content) tuples.
        """
        lines = text.split('\n')
        sections = []
        current_header = "PREAMBLE"
        current_content = []

        for line in lines:
            if self._is_section_boundary(line):
                # Save previous section
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                current_header = line.strip()[:100]  # Limit header length
                current_content = [line]
            else:
                current_content.append(line)

        # Add last section
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))

        return sections if sections else [("DOCUMENT", text)]

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Contract-aware chunking with overlap for obligation tracking.

        Strategy:
        1. First split by contract sections (ARTICLE, SECTION, etc.)
        2. Then apply semantic chunking within sections
        3. Add overlap between chunks to preserve context at boundaries

        Args:
            text: Full document text

        Returns:
            List of chunk dictionaries with text, metadata, and section info
        """
        if not text or not text.strip():
            return []

        # Step 1: Split into logical contract sections
        sections = self._split_into_sections(text)

        all_chunks = []
        chunk_index = 0
        global_char_offset = 0

        for section_header, section_content in sections:
            section_chunks = self._chunk_section_with_overlap(
                section_content,
                section_header,
                chunk_index,
                global_char_offset
            )

            for chunk in section_chunks:
                chunk["section"] = section_header
                all_chunks.append(chunk)
                chunk_index += 1

            global_char_offset += len(section_content) + 1

        logger.info(f"Contract-aware chunking: {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks

    def _chunk_section_with_overlap(
        self,
        section_text: str,
        section_header: str,  # noqa: ARG002 - used for logging context
        start_index: int,
        char_offset: int
    ) -> List[Dict[str, Any]]:
        """
        Chunk a section with overlap to preserve context at boundaries.
        Uses semantic similarity when possible, falls back to size-based chunking.
        """
        sentences = self._split_into_sentences(section_text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [{
                "index": start_index,
                "text": sentences[0],
                "start_char": char_offset,
                "end_char": char_offset + len(sentences[0]),
                "has_overlap": False
            }]

        # Try semantic chunking first
        try:
            embeddings = self.embedding_model.embed_documents(sentences)
            return self._semantic_chunk_with_overlap(
                sentences, embeddings, start_index, char_offset
            )
        except Exception as e:
            logger.warning(f"Semantic chunking failed, using size-based: {e}")
            return self._size_based_chunk_with_overlap(
                sentences, start_index, char_offset
            )

    def _semantic_chunk_with_overlap(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
        start_index: int,
        char_offset: int
    ) -> List[Dict[str, Any]]:
        """Semantic chunking with overlap between chunks."""
        chunks = []
        current_sentences = [sentences[0]]
        current_embedding = embeddings[0]
        chunk_start = char_offset
        overlap_sentences = []  # Sentences to carry over for overlap

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(current_embedding, embeddings[i])
            current_text = ' '.join(current_sentences)

            # Check if we should start a new chunk
            should_split = (
                similarity < self.similarity_threshold or
                len(current_text) + len(sentences[i]) > self.max_chunk_size or
                self._is_section_boundary(sentences[i])
            )

            if should_split and current_sentences:
                chunk_text = ' '.join(current_sentences)
                chunks.append({
                    "index": start_index + len(chunks),
                    "text": chunk_text,
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(chunk_text),
                    "has_overlap": len(overlap_sentences) > 0
                })

                # Calculate overlap: take last N characters worth of sentences
                overlap_sentences = self._get_overlap_sentences(current_sentences)
                chunk_start = chunk_start + len(chunk_text) - len(' '.join(overlap_sentences))

                # Start new chunk with overlap + new sentence
                current_sentences = overlap_sentences + [sentences[i]]
                current_embedding = embeddings[i]
            else:
                current_sentences.append(sentences[i])
                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0).tolist()

        # Add last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append({
                "index": start_index + len(chunks),
                "text": chunk_text,
                "start_char": chunk_start,
                "end_char": chunk_start + len(chunk_text),
                "has_overlap": len(overlap_sentences) > 0
            })

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap, respecting chunk_overlap size."""
        overlap = []
        total_len = 0

        for sentence in reversed(sentences):
            if total_len + len(sentence) > self.chunk_overlap:
                break
            overlap.insert(0, sentence)
            total_len += len(sentence) + 1  # +1 for space

        return overlap

    def _size_based_chunk_with_overlap(
        self,
        sentences: List[str],
        start_index: int,
        char_offset: int
    ) -> List[Dict[str, Any]]:
        """Size-based chunking with overlap when semantic chunking fails."""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = char_offset
        overlap_sentences = []

        for sentence in sentences:
            # Check if adding this sentence exceeds max size
            if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "index": start_index + len(chunks),
                    "text": chunk_text,
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(chunk_text),
                    "has_overlap": len(overlap_sentences) > 0
                })

                # Calculate overlap for next chunk
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                chunk_start = chunk_start + len(chunk_text) - len(' '.join(overlap_sentences))
                current_chunk = overlap_sentences.copy()
                current_size = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += len(sentence)

        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "index": start_index + len(chunks),
                "text": chunk_text,
                "start_char": chunk_start,
                "end_char": chunk_start + len(chunk_text),
                "has_overlap": len(overlap_sentences) > 0
            })

        return chunks

    def _fallback_chunk(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Legacy fallback for backward compatibility."""
        return self._size_based_chunk_with_overlap(sentences, 0, 0)
    
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

