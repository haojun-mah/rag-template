from fastembed import TextEmbedding
from typing import List
from qdrant_client import QdrantClient, models

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

client = QdrantClient(":memory:") # Change this to Qdrant instance else where for production
COLLECTION_NAME = "financial_reports"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=embedding_model.embedding_size,
        distance=models.Distance.COSINE
    )
)

print(f"Qdrant collection '{COLLECTION_NAME}' created.")


def create_embedding_from_chunks(chunks : List[dict]) -> str:
    """Generates embedding from enriched chunk using fast_embed."""

    texts_to_embed = []
    chunks_to_upsert = []

    for i, chunk in enumerate(chunks):
        summary = chunk.get('summary', '')
        keywords = chunk.get('keywords', [])

        # Check if enrichment was skipped (based on the default message set in main.py)
        if summary == "No summary available (enrichment skipped)":
            string_to_embed = f"Content: {chunk['content'][:1000]}"
        else:
            string_to_embed = f"""
            Summary: {summary}
            Keywords: {', '.join(keywords)}
            Content: {chunk['content'][:1000]}
            """
        
        texts_to_embed.append(string_to_embed.strip())
        chunks_to_upsert.append(models.PointStruct(
            id=i,
            vector=[],
            payload=chunk
        ))

    print(f"Prepared chunk {i+1} for embedding.")

    embeddings = list(embedding_model.embed(texts_to_embed, batch_size=32))

    print(f"Embeddings completed. Sample embedding {embeddings[0][:5]}. Upserting into Qdrant.")

    for i, embedding in enumerate(embeddings):
      chunks_to_upsert[i].vector = embedding.tolist()

    client.upsert(
      collection_name=COLLECTION_NAME,
      points=chunks_to_upsert,
      wait=True,
    )

    print("Upsert into Qdrant completed.")

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' info: {collection_info}")

    