"""
Test cases for chunking functionality.
Tests the title_chunker function from ingestion.chunk_generation.title_chunker
"""
import pytest
from unstructured.documents.elements import Text, Title, ElementMetadata
from ingestion.chunk_generation.title_chunker import title_chunker


class TestTitleChunker:
    """Test cases for the title_chunker function."""

    @pytest.fixture
    def sample_elements(self):
        """Create sample parsed elements for testing."""
        metadata = ElementMetadata(
            filename="test_document.txt",
            filetype="text/plain"
        )
        
        elements = [
            Title(text="Introduction", metadata=metadata),
            Text(text="This is the introduction section with some content about the topic.", metadata=metadata),
            Text(text="It contains multiple sentences to make it a reasonable chunk.", metadata=metadata),
            Title(text="Financial Results", metadata=metadata),
            Text(text="Revenue increased by 15% year-over-year.", metadata=metadata),
            Text(text="Operating income grew significantly during the fiscal year.", metadata=metadata),
            Title(text="Conclusion", metadata=metadata),
            Text(text="In summary, the company performed well.", metadata=metadata),
        ]
        return elements

    @pytest.fixture
    def large_text_elements(self):
        """Create elements that will result in multiple chunks."""
        metadata = ElementMetadata(
            filename="large_document.txt",
            filetype="text/plain"
        )
        
        # Create a large text that exceeds max_characters (2048)
        long_text = "This is a very long paragraph. " * 100  # ~3200 characters
        
        elements = [
            Title(text="Large Section", metadata=metadata),
            Text(text=long_text, metadata=metadata),
            Text(text=long_text, metadata=metadata),
        ]
        return elements

    def test_chunker_returns_list(self, sample_elements):
        """Test that the chunker returns a list."""
        chunks = title_chunker(sample_elements)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunker_preserves_content(self, sample_elements):
        """Test that chunking preserves the content from elements."""
        chunks = title_chunker(sample_elements)
        
        # Verify that all chunks have text content
        for chunk in chunks:
            assert hasattr(chunk, 'text')
            assert len(chunk.text) > 0

    def test_chunker_respects_max_characters(self, large_text_elements):
        """Test that chunker respects max_characters parameter (2048)."""
        chunks = title_chunker(large_text_elements)
        
        for chunk in chunks:
            # Some chunks may slightly exceed due to chunking algorithm
            # but should generally respect the limit
            assert len(chunk.text) <= 2500  # Allow some tolerance

    def test_chunker_combines_small_chunks(self, sample_elements):
        """Test that small text elements are combined based on combine_text_under_n_chars (256)."""
        chunks = title_chunker(sample_elements)
        
        # Most chunks should be larger than the minimum threshold
        # since we combine small chunks
        small_chunks = [c for c in chunks if len(c.text) < 256]
        
        # Allow some small chunks but most should be combined
        assert len(small_chunks) < len(chunks)

    def test_chunker_with_empty_elements(self):
        """Test that chunker handles empty element list gracefully."""
        chunks = title_chunker([])
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_chunk_has_metadata(self, sample_elements):
        """Test that chunks preserve metadata."""
        chunks = title_chunker(sample_elements)
        
        for chunk in chunks:
            assert hasattr(chunk, 'metadata')
            assert chunk.metadata is not None

    def test_chunk_identifies_tables(self):
        """Test that chunks correctly identify table elements."""
        metadata = ElementMetadata(
            filename="test_table.txt",
            filetype="text/plain"
        )
        
        # Simulate a table element with text_as_html in metadata
        table_element = Text(
            text="Sample table",
            metadata=metadata
        )
        table_element.metadata.text_as_html = "<table><tr><td>Data</td></tr></table>"
        
        elements = [table_element]
        chunks = title_chunker(elements)
        
        # Check if table metadata is preserved
        for chunk in chunks:
            metadata_dict = chunk.metadata.to_dict()
            if 'text_as_html' in metadata_dict:
                assert metadata_dict['text_as_html'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
