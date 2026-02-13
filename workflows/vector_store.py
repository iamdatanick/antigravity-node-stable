import chromadb


class VectorStore:
    def __init__(self):
        # Professional SRE persistent storage mapping
        self.client = chromadb.PersistentClient(path="/app/data/chroma")
        self.collection = self.client.get_or_create_collection("antigravity_docs")

    async def add_documents(self, chunks, metadatas, ids):
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)

    async def search(self, query, top_k=3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results["documents"][0]

vector_store = VectorStore()
