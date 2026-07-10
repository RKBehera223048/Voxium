from __future__ import annotations

import os
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional

import chromadb

logger = logging.getLogger(__name__)

class VectorStore:
    """
    A persistent vector store using ChromaDB.

    This class handles connecting to a ChromaDB persistent client, creating or
    retrieving collections, and provides asynchronous methods to add chunks,
    search by vector, and delete documents.
    """

    def __init__(self, collection_name: str = "voxium_memory", db_path: Optional[str] = None):
        """
        Initializes the VectorStore with a specific collection name.

        Args:
            collection_name (str): The name of the ChromaDB collection to use.
            db_path (str): The persistent directory for ChromaDB.
        """
        self.collection_name = collection_name
        self.persist_directory = db_path or os.path.join(os.getcwd(), 'data', 'db', 'chromadb')
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB PersistentClient at {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.debug(f"Loaded/created collection: {self.collection_name}")

    async def add_chunks(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """
        Adds text chunks, their embeddings, and metadata to the vector store.

        Args:
            texts (List[str]): List of string texts to store.
            embeddings (List[List[float]]): List of embeddings corresponding to the texts.
            metadatas (List[dict]): List of metadata dictionaries corresponding to the texts.
        """
        if not texts:
            logger.warning("Empty list of texts provided to add_chunks.")
            return

        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("Lengths of texts, embeddings, and metadatas must match.")

        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in texts]

        def _add():
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        logger.debug(f"Adding {len(texts)} chunks to collection {self.collection_name}.")
        await asyncio.to_thread(_add)
        logger.info(f"Successfully added {len(texts)} chunks to VectorStore.")

    async def search(self, query_embedding: List[float], n_results: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Searches the vector store for the closest chunks to the query_embedding.

        Args:
            query_embedding (List[float]): The embedding vector to search with.
            n_results (int): The maximum number of results to return.
            filter_dict (Optional[dict]): ChromaDB metadata filter dictionary.

        Returns:
            List[dict]: A list of dictionaries, each containing 'text', 'metadata', and 'distance'.
        """
        def _search():
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict
            )

        logger.debug(f"Searching VectorStore for {n_results} results with filter: {filter_dict}")
        results = await asyncio.to_thread(_search)

        parsed_results = []
        if not results or not results.get('documents') or not results['documents'][0]:
            logger.debug("No results found in search.")
            return parsed_results

        # Chroma returns lists of lists for documents, metadatas, distances, etc.
        # Since we query with one embedding, we access the first list in each.
        docs = results['documents'][0]
        metadatas = results['metadatas'][0] if results.get('metadatas') else [None] * len(docs)
        distances = results['distances'][0] if results.get('distances') else [None] * len(docs)

        for doc, metadata, distance in zip(docs, metadatas, distances):
            parsed_results.append({
                'text': doc,
                'metadata': metadata,
                'distance': distance
            })

        logger.info(f"Search completed. Found {len(parsed_results)} results.")
        return parsed_results

    async def delete(self, ids: List[str]) -> None:
        """
        Deletes documents from the vector store by their IDs.

        Args:
            ids (List[str]): List of string IDs of the documents to delete.
        """
        if not ids:
            logger.warning("Empty list of IDs provided to delete.")
            return

        def _delete():
            self.collection.delete(ids=ids)

        logger.debug(f"Deleting {len(ids)} documents from collection {self.collection_name}.")
        await asyncio.to_thread(_delete)
        logger.info(f"Successfully deleted {len(ids)} documents from VectorStore.")
