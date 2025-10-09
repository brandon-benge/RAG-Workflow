from __future__ import annotations

def get_embedder(model: str):
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder  # type: ignore
    return SentenceTransformersDocumentEmbedder(model=model or 'sentence-transformers/all-MiniLM-L6-v2')

def write_embeddings(document_store, embedder, docs):
    """
    Embed documents and return them WITHOUT writing to the store.
    Caller is responsible for metadata sanitation and final write.
    """
    embedder.warm_up()
    embedded_docs = embedder.run(documents=docs)["documents"]
    return embedded_docs
