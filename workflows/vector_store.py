
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, host="chromadb", port=8000):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection("antigravity")

    def add_documents(self, documents: list, ids: list):
        self.collection.add(documents=documents, ids=ids)

    def query(self, text: str, n_results: int = 3):
        return self.collection.query(query_texts=[text], n_results=n_results)
