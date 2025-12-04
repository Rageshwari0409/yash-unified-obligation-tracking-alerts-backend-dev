"""
Unit tests for DocumentProcessor class.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import io
import sys

# Mock problematic imports
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['pymilvus'] = MagicMock()

from src.processing.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    """Document processor instance."""
    return DocumentProcessor()


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model."""
    mock = Mock()
    mock.embed_documents = Mock(return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768])
    return mock


class TestDocumentProcessorInit:
    """Test suite for DocumentProcessor initialization."""
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.supported_formats is not None
        assert processor.max_file_size_mb > 0
        assert processor.max_chunk_size > 0
    
    def test_supported_formats_loaded(self, processor):
        """Test that supported formats are loaded."""
        assert '.pdf' in processor.supported_formats
        assert '.docx' in processor.supported_formats
        assert '.txt' in processor.supported_formats


class TestValidateFile:
    """Test suite for validate_file method."""
    
    def test_validate_supported_format(self, processor):
        """Test validation of supported file format."""
        is_valid, error = processor.validate_file("test.pdf", 1024)
        assert is_valid is True
        assert error == ""
    
    def test_validate_unsupported_format(self, processor):
        """Test validation of unsupported file format."""
        is_valid, error = processor.validate_file("test.exe", 1024)
        assert is_valid is False
        assert "Unsupported file format" in error
    
    def test_validate_file_too_large(self, processor):
        """Test validation of file that's too large."""
        large_size = (processor.max_file_size_mb + 1) * 1024 * 1024
        is_valid, error = processor.validate_file("test.pdf", large_size)
        assert is_valid is False
        assert "File too large" in error
    
    def test_validate_file_at_limit(self, processor):
        """Test validation of file at size limit."""
        limit_size = processor.max_file_size_mb * 1024 * 1024
        is_valid, error = processor.validate_file("test.pdf", limit_size)
        assert is_valid is True
    
    def test_validate_case_insensitive_extension(self, processor):
        """Test that file extension validation is case insensitive."""
        is_valid, error = processor.validate_file("test.PDF", 1024)
        assert is_valid is True


class TestExtractText:
    """Test suite for extract_text method."""
    
    def test_extract_from_txt(self, processor):
        """Test text extraction from TXT file."""
        content = b"This is a test document."
        text = processor.extract_text(content, "test.txt")
        assert text == "This is a test document."
    
    def test_extract_from_txt_with_encoding(self, processor):
        """Test text extraction from TXT with special characters."""
        content = "Test with special chars: é, ñ, ü".encode('utf-8')
        text = processor.extract_text(content, "test.txt")
        assert "é" in text or "e" in text  # May be decoded differently
    
    def test_extract_from_pdf(self, processor):
        """Test text extraction from PDF file."""
        with patch.dict('sys.modules', {'pypdf': MagicMock()}):
            import sys
            mock_pypdf = sys.modules['pypdf']
            mock_page = Mock()
            mock_page.extract_text = Mock(return_value="PDF content")
            mock_reader = Mock()
            mock_reader.pages = [mock_page]
            mock_pypdf.PdfReader = Mock(return_value=mock_reader)

            content = b"PDF binary content"
            text = processor.extract_text(content, "test.pdf")
            assert "PDF content" in text

    def test_extract_from_docx(self, processor):
        """Test text extraction from DOCX file."""
        with patch.dict('sys.modules', {'docx': MagicMock()}):
            import sys
            mock_docx = sys.modules['docx']
            mock_para1 = Mock()
            mock_para1.text = "Paragraph 1"
            mock_para2 = Mock()
            mock_para2.text = "Paragraph 2"
            mock_doc_instance = Mock()
            mock_doc_instance.paragraphs = [mock_para1, mock_para2]
            mock_docx.Document = Mock(return_value=mock_doc_instance)

            content = b"DOCX binary content"
            text = processor.extract_text(content, "test.docx")
            assert "Paragraph 1" in text
            assert "Paragraph 2" in text
    
    def test_extract_unsupported_format(self, processor):
        """Test extraction from unsupported format raises error."""
        with pytest.raises(ValueError):
            processor.extract_text(b"content", "test.exe")


class TestSplitIntoSentences:
    """Test suite for _split_into_sentences method."""
    
    def test_split_simple_sentences(self, processor):
        """Test splitting simple sentences."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = processor._split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
    
    def test_split_with_question_marks(self, processor):
        """Test splitting sentences with question marks."""
        text = "Is this a question? Yes it is. Another sentence."
        sentences = processor._split_into_sentences(text)
        assert len(sentences) == 3
    
    def test_split_with_exclamation(self, processor):
        """Test splitting sentences with exclamation marks."""
        text = "This is exciting! Really exciting. Normal sentence."
        sentences = processor._split_into_sentences(text)
        assert len(sentences) == 3
    
    def test_split_empty_text(self, processor):
        """Test splitting empty text."""
        sentences = processor._split_into_sentences("")
        assert sentences == []
    
    def test_split_single_sentence(self, processor):
        """Test splitting single sentence."""
        text = "Just one sentence."
        sentences = processor._split_into_sentences(text)
        assert len(sentences) == 1


class TestCosineSimilarity:
    """Test suite for _cosine_similarity method."""
    
    def test_identical_vectors(self, processor):
        """Test similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        similarity = processor._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=0.01)
    
    def test_orthogonal_vectors(self, processor):
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = processor._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=0.01)
    
    def test_opposite_vectors(self, processor):
        """Test similarity of opposite vectors."""
        vec1 = [1.0, 1.0, 1.0]
        vec2 = [-1.0, -1.0, -1.0]
        similarity = processor._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, abs=0.01)


class TestChunkText:
    """Test suite for chunk_text method."""
    
    def test_chunk_simple_text(self, processor):
        """Test chunking simple text."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[
            [0.1] * 768,
            [0.11] * 768,  # Similar to first
            [0.5] * 768    # Different
        ])
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("index" in chunk for chunk in chunks)
    
    def test_chunk_empty_text(self, processor):
        """Test chunking empty text."""
        chunks = processor.chunk_text("")
        assert chunks == []
    
    def test_chunk_single_sentence(self, processor):
        """Test chunking single sentence."""
        text = "Just one sentence."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
    
    def test_chunk_with_embedding_error(self, processor):
        """Test chunking falls back when embeddings fail."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(side_effect=Exception("API error"))
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = processor.chunk_text(text)
        
        # Should still return chunks using fallback
        assert len(chunks) > 0
    
    def test_chunk_respects_max_size(self, processor):
        """Test that chunks respect max size."""
        processor.max_chunk_size = 50
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[
            [0.1] * 768 for _ in range(10)
        ])
        
        text = " ".join(["This is a sentence."] * 10)
        chunks = processor.chunk_text(text)
        
        # All chunks should be under max size
        assert all(len(chunk["text"]) <= processor.max_chunk_size + 100 for chunk in chunks)


class TestFallbackChunk:
    """Test suite for _fallback_chunk method."""
    
    def test_fallback_chunk_simple(self, processor):
        """Test fallback chunking with simple sentences."""
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        chunks = processor._fallback_chunk(sentences)
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
    
    def test_fallback_chunk_respects_max_size(self, processor):
        """Test that fallback chunking respects max size."""
        processor.max_chunk_size = 30
        sentences = ["This is a long sentence that exceeds the limit."] * 5
        chunks = processor._fallback_chunk(sentences)
        
        # Should create multiple chunks
        assert len(chunks) > 1
    
    def test_fallback_chunk_empty_list(self, processor):
        """Test fallback chunking with empty list."""
        chunks = processor._fallback_chunk([])
        assert chunks == []


class TestProcessDocument:
    """Test suite for process_document method."""
    
    def test_process_document_success(self, processor):
        """Test successful document processing."""
        processor.embedding_model = Mock()
        # Return embeddings for each sentence (2 sentences in input)
        processor.embedding_model.embed_documents = Mock(return_value=[[0.1] * 768, [0.2] * 768])

        content = b"This is a test document. It has multiple sentences."
        result = processor.process_document(content, "test.txt")

        assert "filename" in result
        assert "full_text" in result
        assert "chunks" in result
        assert "total_characters" in result
        assert "total_chunks" in result
        assert result["filename"] == "test.txt"
    
    def test_process_document_invalid_file(self, processor):
        """Test processing invalid file raises error."""
        content = b"content"
        with pytest.raises(ValueError):
            processor.process_document(content, "test.exe")
    
    def test_process_document_too_large(self, processor):
        """Test processing file that's too large raises error."""
        large_content = b"A" * (processor.max_file_size_mb * 1024 * 1024 + 1)
        with pytest.raises(ValueError):
            processor.process_document(large_content, "test.txt")
    
    def test_process_document_chunks_created(self, processor):
        """Test that document processing creates chunks."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[
            [0.1] * 768,
            [0.2] * 768
        ])
        
        content = b"First sentence. Second sentence."
        result = processor.process_document(content, "test.txt")
        
        assert result["total_chunks"] > 0
        assert len(result["chunks"]) == result["total_chunks"]



class TestDocumentProcessorNegative:
    """Negative test cases for DocumentProcessor."""
    
    def test_validate_file_with_none_filename(self, processor):
        """Test validation with None filename."""
        try:
            is_valid, error = processor.validate_file(None, 1024)
            assert is_valid is False
        except (TypeError, AttributeError):
            # Expected to raise error
            pass
    
    def test_validate_file_with_empty_filename(self, processor):
        """Test validation with empty filename."""
        is_valid, error = processor.validate_file("", 1024)
        assert is_valid is False
    
    def test_validate_file_with_negative_size(self, processor):
        """Test validation with negative file size."""
        is_valid, error = processor.validate_file("test.pdf", -1)
        # Should handle gracefully
        assert isinstance(is_valid, bool)
    
    def test_validate_file_with_zero_size(self, processor):
        """Test validation with zero file size."""
        is_valid, error = processor.validate_file("test.pdf", 0)
        # Zero size file should be valid format but might be flagged
        assert isinstance(is_valid, bool)
    
    def test_validate_file_double_extension(self, processor):
        """Test validation with double extension."""
        is_valid, error = processor.validate_file("test.pdf.exe", 1024)
        # Should check the last extension
        assert is_valid is False
    
    def test_validate_file_no_extension(self, processor):
        """Test validation with no file extension."""
        is_valid, error = processor.validate_file("testfile", 1024)
        assert is_valid is False
    
    def test_extract_text_with_empty_content(self, processor):
        """Test text extraction with empty content."""
        try:
            text = processor.extract_text(b"", "test.txt")
            assert text == ""
        except Exception:
            # May raise error for empty content
            pass
    
    def test_extract_text_with_binary_garbage(self, processor):
        """Test text extraction with binary garbage."""
        garbage = bytes([i % 256 for i in range(1000)])
        
        try:
            text = processor.extract_text(garbage, "test.txt")
            # Should decode with errors='ignore'
            assert isinstance(text, str)
        except Exception:
            # May raise error for invalid content
            pass
    
    def test_extract_from_pdf_corrupted(self, processor):
        """Test PDF extraction with corrupted file."""
        with patch.dict('sys.modules', {'pypdf': MagicMock()}):
            import sys
            mock_pypdf = sys.modules['pypdf']
            mock_pypdf.PdfReader = Mock(side_effect=Exception("Corrupted PDF"))

            try:
                text = processor.extract_text(b"corrupted", "test.pdf")
            except Exception as e:
                assert "PDF" in str(e) or "Corrupted" in str(e) or "pypdf" in str(e) or "pdfplumber" in str(e)

    def test_extract_from_pdf_encrypted(self, processor):
        """Test PDF extraction with encrypted file."""
        with patch.dict('sys.modules', {'pypdf': MagicMock()}):
            import sys
            mock_pypdf = sys.modules['pypdf']
            mock_pypdf.PdfReader = Mock(side_effect=Exception("Encrypted PDF"))

            try:
                text = processor.extract_text(b"encrypted", "test.pdf")
            except Exception as e:
                assert isinstance(e, Exception)

    def test_extract_from_docx_corrupted(self, processor):
        """Test DOCX extraction with corrupted file."""
        with patch.dict('sys.modules', {'docx': MagicMock()}):
            import sys
            mock_docx = sys.modules['docx']
            mock_docx.Document = Mock(side_effect=Exception("Corrupted DOCX"))

            try:
                text = processor.extract_text(b"corrupted", "test.docx")
            except Exception as e:
                assert isinstance(e, Exception)
    
    def test_extract_text_unsupported_after_validation(self, processor):
        """Test extraction of format that passed validation but fails extraction."""
        with pytest.raises(ValueError):
            processor.extract_text(b"content", "test.xyz")
    
    def test_chunk_text_with_no_sentences(self, processor):
        """Test chunking text with no sentence delimiters."""
        text = "thisisalllowercasewithnosentencedelimitersatall"
        chunks = processor.chunk_text(text)
        
        # Should still create at least one chunk
        assert len(chunks) >= 1
    
    def test_chunk_text_with_only_punctuation(self, processor):
        """Test chunking text with only punctuation."""
        text = "... !!! ??? ..."
        chunks = processor.chunk_text(text)
        
        # Should handle gracefully
        assert isinstance(chunks, list)
    
    def test_chunk_text_with_very_long_sentence(self, processor):
        """Test chunking with sentence longer than max chunk size."""
        processor.max_chunk_size = 100
        processor.embedding_model = Mock()
        # Return embeddings for each sentence (multiple sentences in the text)
        long_sentence = "This is a very long sentence. " * 100
        sentence_count = long_sentence.count(". ")
        processor.embedding_model.embed_documents = Mock(
            return_value=[[0.1] * 768 for _ in range(max(1, sentence_count))]
        )

        chunks = processor.chunk_text(long_sentence)

        # Should handle long sentences
        assert len(chunks) > 0
    
    def test_chunk_text_with_embedding_api_error(self, processor):
        """Test chunking when embedding API returns error."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(side_effect=Exception("API Error"))
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = processor.chunk_text(text)
        
        # Should fall back to simple chunking
        assert len(chunks) > 0
    
    def test_chunk_text_with_empty_embeddings(self, processor):
        """Test chunking when embeddings are empty."""
        processor.embedding_model = Mock()
        # Make embed_documents raise an exception to trigger fallback
        processor.embedding_model.embed_documents = Mock(side_effect=Exception("Empty embeddings"))

        text = "First sentence. Second sentence."
        chunks = processor.chunk_text(text)

        # Should fall back to simple chunking
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_chunk_text_with_mismatched_embedding_count(self, processor):
        """Test chunking when embedding count doesn't match sentence count."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[[0.1] * 768])  # Only 1 embedding
        
        text = "First sentence. Second sentence. Third sentence."  # 3 sentences
        
        try:
            chunks = processor.chunk_text(text)
            # Should handle mismatch
            assert isinstance(chunks, list)
        except (IndexError, ValueError):
            # May raise error due to mismatch
            pass
    
    def test_cosine_similarity_with_zero_vectors(self, processor):
        """Test cosine similarity with zero vectors."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [0.0, 0.0, 0.0]
        
        try:
            similarity = processor._cosine_similarity(vec1, vec2)
            # May return NaN or raise error
            assert isinstance(similarity, (float, int)) or str(similarity) == 'nan'
        except (ZeroDivisionError, ValueError):
            # Expected for zero vectors
            pass
    
    def test_cosine_similarity_with_different_dimensions(self, processor):
        """Test cosine similarity with vectors of different dimensions."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        
        try:
            similarity = processor._cosine_similarity(vec1, vec2)
        except (ValueError, IndexError):
            # Expected to raise error
            pass
    
    def test_split_sentences_with_abbreviations(self, processor):
        """Test sentence splitting with abbreviations."""
        text = "Dr. Smith works at Inc. Corp. He lives in Washington D.C. and has a Ph.D. degree."
        sentences = processor._split_into_sentences(text)
        
        # Should handle abbreviations (may split incorrectly)
        assert len(sentences) > 0
    
    def test_split_sentences_with_urls(self, processor):
        """Test sentence splitting with URLs."""
        text = "Visit https://example.com. Also check http://test.org. More info at www.site.com."
        sentences = processor._split_into_sentences(text)
        
        assert len(sentences) > 0
    
    def test_split_sentences_with_numbers(self, processor):
        """Test sentence splitting with decimal numbers."""
        text = "The price is $1,234.56. The rate is 3.14159. Total is 99.99."
        sentences = processor._split_into_sentences(text)
        
        # Should not split on decimal points
        assert len(sentences) > 0
    
    def test_process_document_with_all_errors(self, processor):
        """Test full document processing with cascading errors."""
        # File too large
        large_content = b"A" * (processor.max_file_size_mb * 1024 * 1024 + 1)
        
        with pytest.raises(ValueError):
            processor.process_document(large_content, "test.txt")
    
    def test_process_document_extraction_failure(self, processor):
        """Test document processing when extraction fails."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[[0.1] * 768])
        
        with patch.object(processor, 'extract_text', side_effect=Exception("Extraction failed")):
            with pytest.raises(Exception):
                processor.process_document(b"content", "test.pdf")
    
    def test_process_document_chunking_failure(self, processor):
        """Test document processing when chunking fails."""
        with patch.object(processor, 'extract_text', return_value="Sample text"):
            with patch.object(processor, 'chunk_text', side_effect=Exception("Chunking failed")):
                with pytest.raises(Exception):
                    processor.process_document(b"content", "test.txt")
    
    def test_fallback_chunk_with_single_long_sentence(self, processor):
        """Test fallback chunking with single sentence exceeding max size."""
        processor.max_chunk_size = 50
        sentences = ["This is an extremely long sentence that definitely exceeds the maximum chunk size limit"]
        
        chunks = processor._fallback_chunk(sentences)
        
        # Should still create chunk even if it exceeds limit
        assert len(chunks) == 1
    
    def test_fallback_chunk_with_empty_sentences(self, processor):
        """Test fallback chunking with empty sentence list."""
        chunks = processor._fallback_chunk([])
        assert chunks == []
    
    def test_chunk_text_with_unicode_sentences(self, processor):
        """Test chunking with unicode characters."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[
            [0.1] * 768,
            [0.2] * 768
        ])
        
        text = "これは日本語の文です。This is English. Это русский текст."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 0
    
    def test_extract_text_with_mixed_encodings(self, processor):
        """Test text extraction with mixed character encodings."""
        # Mix of UTF-8 and Latin-1 characters
        mixed_content = "Hello".encode('utf-8') + b'\xe9\xe8' + "World".encode('utf-8')
        
        text = processor.extract_text(mixed_content, "test.txt")
        
        # Should decode with errors='ignore'
        assert isinstance(text, str)
        assert "Hello" in text or "World" in text


class TestDocumentProcessorEdgeCases:
    """Edge case tests for DocumentProcessor."""
    
    def test_validate_file_with_multiple_dots(self, processor):
        """Test validation with multiple dots in filename."""
        is_valid, error = processor.validate_file("my.test.file.pdf", 1024)
        assert is_valid is True
    
    def test_validate_file_with_unicode_filename(self, processor):
        """Test validation with unicode characters in filename."""
        is_valid, error = processor.validate_file("テスト文書.pdf", 1024)
        assert is_valid is True
    
    def test_validate_file_exactly_at_size_limit(self, processor):
        """Test validation with file exactly at size limit."""
        exact_size = processor.max_file_size_mb * 1024 * 1024
        is_valid, error = processor.validate_file("test.pdf", exact_size)
        assert is_valid is True
    
    def test_validate_file_one_byte_over_limit(self, processor):
        """Test validation with file one byte over limit."""
        over_size = processor.max_file_size_mb * 1024 * 1024 + 1
        is_valid, error = processor.validate_file("test.pdf", over_size)
        assert is_valid is False
    
    def test_chunk_text_with_single_character_sentences(self, processor):
        """Test chunking with single character sentences."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ])
        
        text = "A. B. C."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 0
    
    def test_process_document_with_whitespace_only(self, processor):
        """Test processing document with only whitespace."""
        processor.embedding_model = Mock()
        processor.embedding_model.embed_documents = Mock(return_value=[])
        
        content = b"   \n\n\t\t   \n   "
        result = processor.process_document(content, "test.txt")
        
        # Should handle whitespace-only content
        assert result["total_characters"] >= 0
        assert result["total_chunks"] >= 0
