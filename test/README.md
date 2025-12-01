# RAG Pipeline Tests

This directory contains test cases for the RAG (Retrieval-Augmented Generation) pipeline components.

## Test Structure

- `test_chunking.py` - Tests for document chunking functionality
- `test_enrichment.py` - Tests for chunk enrichment with LLM-generated metadata
- `conftest.py` - Shared pytest configuration and fixtures

## Running Tests

### Run all tests
```bash
uv run pytest test/
```

### Run tests with verbose output
```bash
uv run pytest test/ -v
```

### Run specific test file
```bash
uv run pytest test/test_chunking.py -v
uv run pytest test/test_enrichment.py -v
```

### Run specific test class or method
```bash
uv run pytest test/test_chunking.py::TestTitleChunker -v
uv run pytest test/test_enrichment.py::TestChunkEnrichment::test_enrich_chunk_returns_dict -v
```

### Run with coverage
```bash
uv run pytest test/ --cov=ingestion --cov=llm --cov-report=html
```

## Test Coverage

### Chunking Tests (`test_chunking.py`)
- ✅ Chunker returns a list of chunks
- ✅ Content is preserved during chunking
- ✅ Respects max_characters parameter (2048)
- ✅ Combines small chunks based on combine_text_under_n_chars (256)
- ✅ Handles empty element lists gracefully
- ✅ Preserves metadata in chunks
- ✅ Correctly identifies table elements

### Enrichment Tests (`test_enrichment.py`)
- ✅ Returns dictionary with enriched metadata
- ✅ Contains all required fields (summary, keywords, hypothetical_questions, table_summary)
- ✅ Summary is not empty for text chunks
- ✅ Keywords is a list with appropriate items (5-7 expected)
- ✅ Hypothetical questions is a list (3-5 expected)
- ✅ Table summary is None for non-table chunks
- ✅ Table summary is populated for table chunks
- ✅ Truncates content to 3000 characters before enrichment
- ✅ Handles errors gracefully by returning empty dict
- ✅ Correctly identifies table vs text chunks

## Dependencies

The tests use:
- `pytest` - Test framework
- `unittest.mock` - Mocking for LLM API calls
- `unstructured` - Document elements for test fixtures

## Environment

Tests automatically load environment variables from the `.env` file in the project root via `conftest.py`.
