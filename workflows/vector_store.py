import logging
import os

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger("vector-store")

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))


def get_vector_client():
    if not chromadb:
        logger.error("chromadb-client not installed")
        return None
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def add_to_index(collection_name: str, ids: list, embeddings: list, metadatas: list, documents: list):
    client = get_vector_client()
    if not client:
        return
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


def search_index(collection_name: str, query_embedding: list, n_results: int = 3):
    client = get_vector_client()
    if not client:
        return {"documents": [[]], "metadatas": [[]]}
    collection = client.get_or_create_collection(name=collection_name)
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)
