"""
Test cases for chunk enrichment functionality.
Tests the enrich_chunk function from ingestion.chunk_enrichment.llm_enrichment
"""
import pytest
from unittest.mock import Mock, patch
from unstructured.documents.elements import Text, ElementMetadata
from ingestion.chunk_enrichment.llm_enrichment import enrich_chunk
from llm.llm_for_chunking import ChunkMetadata


class TestChunkEnrichment:
    """Test cases for the enrich_chunk function."""

    @pytest.fixture
    def sample_text_chunk(self):
        """Create a sample text chunk for testing."""
        metadata = ElementMetadata(
            filename="test_document.txt",
            filetype="text/plain"
        )
        
        chunk = Text(
            text="Microsoft Corporation reported a 15% increase in revenue for fiscal year 2024. The cloud segment was the primary driver of this growth, showing strong performance across all regions.",
            metadata=metadata
        )
        return chunk

    @pytest.fixture
    def sample_table_chunk(self):
        """Create a sample table chunk for testing."""
        metadata = ElementMetadata(
            filename="test_table.txt",
            filetype="text/plain"
        )
        
        chunk = Text(
            text="Revenue table",
            metadata=metadata
        )
        chunk.metadata.text_as_html = """
        <table>
            <tr><th>Quarter</th><th>Revenue</th></tr>
            <tr><td>Q1</td><td>$100M</td></tr>
            <tr><td>Q2</td><td>$115M</td></tr>
        </table>
        """
        return chunk

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        mock_metadata = ChunkMetadata(
            summary="Microsoft saw 15% revenue growth in fiscal 2024, driven by cloud.",
            keywords=["Microsoft", "revenue", "fiscal year 2024", "growth", "cloud segment"],
            hypothetical_questions=[
                "What was Microsoft's revenue growth in fiscal year 2024?",
                "Which segment drove Microsoft's growth?",
                "How did the cloud segment perform in 2024?"
            ],
            table_summary=None
        )
        return mock_metadata

    @pytest.fixture
    def mock_table_llm_response(self):
        """Create a mock LLM response for table data."""
        mock_metadata = ChunkMetadata(
            summary="Quarterly revenue data showing growth from Q1 to Q2.",
            keywords=["revenue", "quarterly", "Q1", "Q2", "growth"],
            hypothetical_questions=[
                "What was the revenue in Q1?",
                "How much did revenue grow from Q1 to Q2?",
                "What is the quarterly revenue trend?"
            ],
            table_summary="Revenue increased from $100M in Q1 to $115M in Q2, representing 15% growth."
        )
        return mock_metadata

    def test_enrich_chunk_returns_dict(self, sample_text_chunk, mock_llm_response):
        """Test that enrich_chunk returns a list of dictionaries."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_llm_response):
            result = enrich_chunk([sample_text_chunk])
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)

    def test_enriched_chunk_has_required_fields(self, sample_text_chunk, mock_llm_response):
        """Test that enriched chunk contains all required metadata fields."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_llm_response):
            result = enrich_chunk([sample_text_chunk])
            
            assert len(result) == 1
            enriched = result[0]
            assert 'summary' in enriched
            assert 'keywords' in enriched
            assert 'hypothetical_questions' in enriched
            assert 'table_summary' in enriched

    def test_enriched_chunk_summary_not_empty(self, sample_text_chunk, mock_llm_response):
        """Test that summary is not empty for text chunks."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_llm_response):
            result = enrich_chunk([sample_text_chunk])
            
            assert len(result) == 1
            enriched = result[0]
            assert enriched['summary'] is not None
            assert len(enriched['summary']) > 0
            assert isinstance(enriched['summary'], str)

    def test_enriched_chunk_keywords_is_list(self, sample_text_chunk, mock_llm_response):
        """Test that keywords is a list."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_llm_response):
            result = enrich_chunk([sample_text_chunk])
            
            assert len(result) == 1
            enriched = result[0]
            assert isinstance(enriched['keywords'], list)
            assert len(enriched['keywords']) > 0
            # Keywords should typically be 5-7 items
            assert len(enriched['keywords']) >= 3

    def test_enriched_chunk_questions_is_list(self, sample_text_chunk, mock_llm_response):
        """Test that hypothetical_questions is a list."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_llm_response):
            result = enrich_chunk([sample_text_chunk])
            
            assert len(result) == 1
            enriched = result[0]
            assert isinstance(enriched['hypothetical_questions'], list)
            assert len(enriched['hypothetical_questions']) > 0
            # Should have 3-5 questions
            assert len(enriched['hypothetical_questions']) >= 3

    def test_text_chunk_table_summary_is_none(self, sample_text_chunk, mock_llm_response):
        """Test that table_summary is None for non-table chunks."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_llm_response):
            result = enrich_chunk([sample_text_chunk])
            
            assert len(result) == 1
            enriched = result[0]
            # Text chunks should have table_summary as None
            assert enriched['table_summary'] is None

    def test_table_chunk_has_table_summary(self, sample_table_chunk, mock_table_llm_response):
        """Test that table chunks have table_summary populated."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_table_llm_response):
            result = enrich_chunk([sample_table_chunk])
            
            assert len(result) == 1
            enriched = result[0]
            assert enriched['table_summary'] is not None
            assert len(enriched['table_summary']) > 0
            assert isinstance(enriched['table_summary'], str)

    def test_enrich_chunk_truncates_long_content(self, sample_text_chunk, mock_llm_response):
        """Test that content is truncated to 3000 characters before enrichment."""
        # Create a chunk with very long text
        long_text = "This is a very long text. " * 200  # ~5000 characters
        sample_text_chunk.text = long_text
        
        with patch('ingestion.chunk_enrichment.llm_enrichment.chunk_enricher') as mock_enricher:
            mock_enricher.return_value = "mocked prompt"
            with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_llm_response):
                enrich_chunk([sample_text_chunk])
                
                # Verify that chunk_enricher was called with truncated content
                call_args = mock_enricher.call_args
                assert call_args is not None
                # First argument should be truncated content
                assert len(call_args[0][0]) <= 3000

    def test_enrich_chunk_handles_error_gracefully(self, sample_text_chunk):
        """Test that enrich_chunk returns empty dict on error."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', side_effect=Exception("API Error")):
            result = enrich_chunk([sample_text_chunk])
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert len(result[0]) == 0  # Should return empty dict on error

    def test_enrich_identifies_table_correctly(self, sample_table_chunk, mock_table_llm_response):
        """Test that enrichment correctly identifies table chunks."""
        with patch('ingestion.chunk_enrichment.llm_enrichment.chunk_enricher') as mock_enricher:
            mock_enricher.return_value = "mocked prompt"
            with patch('ingestion.chunk_enrichment.llm_enrichment.generate_enriched_chunk', return_value=mock_table_llm_response):
                enrich_chunk([sample_table_chunk])
                
                # Verify chunk_enricher was called with is_table=True
                call_args = mock_enricher.call_args
                assert call_args is not None
                # Second argument should be is_table flag
                assert call_args[0][1] is True  # is_table should be True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
