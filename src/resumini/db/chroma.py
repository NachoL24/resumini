import os

import chromadb
from chromadb.utils import embedding_functions


class ChromaClient:
    def __init__(
        self,
        persist_dir: str,
        embedding_model_name: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self._ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=embedding_model_name,
            )
        else:
            self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name="resumini",
            embedding_function=self._ef,
        )

    def index_document(self, doc_id: str, content: str, metadata: dict):
        existing = self._collection.get(ids=[doc_id])
        if existing["ids"]:
            self._collection.update(ids=[doc_id], documents=[content], metadatas=[metadata])
        else:
            self._collection.add(ids=[doc_id], documents=[content], metadatas=[metadata])

    def search(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        kwargs: dict = {"query_texts": [query], "n_results": n_results}
        if metadata_filter:
            kwargs["where"] = metadata_filter
        results = self._collection.query(**kwargs)
        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return docs

    def delete_document(self, doc_id: str):
        self._collection.delete(ids=[doc_id])
