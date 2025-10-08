from __future__ import annotations

def get_embedder(local: bool, model: str):
    if local:
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder  # type: ignore
        return SentenceTransformersDocumentEmbedder(model=model or 'sentence-transformers/all-MiniLM-L6-v2')
    else:
        from haystack.components.embedders import OpenAIDocumentEmbedder  # type: ignore
        return OpenAIDocumentEmbedder(model=model or 'text-embedding-3-small')

def write_embeddings(document_store, embedder, docs):
    """
    Embed documents and return them WITHOUT writing to the store.
    Caller is responsible for metadata sanitation and final write.
    """
    embedder.warm_up()
    embedded_docs = embedder.run(documents=docs)["documents"]
    return embedded_docs
