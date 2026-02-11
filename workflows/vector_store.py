
import chromadb
import os

client = chromadb.HttpClient(
    host=os.environ.get("CHROMA_HOST", "chromadb"),
    port=int(os.environ.get("CHROMA_PORT", 8000))
)

def add_documents(collection_name: str, documents: list[str], embeddings: list[list[float]], ids: list[str]):
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

def query(collection_name: str, query_embedding: list[float], top_k: int = 3):
    collection = client.get_or_create_collection(name=collection_name)
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
