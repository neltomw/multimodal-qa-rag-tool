# utils/vector_store.py
from langchain_community.vectorstores import PGVector
#from langchain.vectorstores import PGVector
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any
import logging

#def initialize_vector_store(connection_string: str, collection_name: str, embedding: Embeddings) -> PGVector:
def initialize_vector_store(connection_string: str, embedding: Embeddings) -> PGVector:

    """
    Initializes and returns a PGVector instance.

    Args:
        connection_string (str): The connection string for the database.
        collection_name (str): The name of the collection in the vector store.
        embedding (Embeddings): The embedding model to use.

    Returns:
        PGVector: An instance of PGVector.
    """
    try:
        return PGVector.from_existing_index(
            embedding=embedding,
        #    collection_name=collection_name,
            connection_string=connection_string
        )
    except Exception as e:
        logging.error(f"Error initializing vector store: {str(e)}")
        raise

def add_texts_to_vector_store(vector_store: PGVector, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
    """
    Adds texts and their metadata to the vector store.

    Args:
        vector_store (PGVector): The vector store to add texts to.
        texts (List[str]): The texts to add.
        metadatas (List[Dict[str, Any]]): The metadata for each text.

    Returns:
        List[str]: The IDs of the added texts in the vector store.
    """
    try:
        return vector_store.add_texts(texts=texts, metadatas=metadatas)
    except Exception as e:
        logging.error(f"Error adding texts to vector store: {str(e)}")
        raise

def similarity_search(vector_store: PGVector, query: str, k: int = 4) -> List[Any]:
    """
    Performs a similarity search in the vector store.

    Args:
        vector_store (PGVector): The vector store to search in.
        query (str): The query text.
        k (int): The number of results to return.

    Returns:
        List[Any]: The k most similar items from the vector store.
    """
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        logging.error(f"Error performing similarity search: {str(e)}")
        raise

# You might add more functions here for other vector store operations