from langchain.embeddings import OpenAIEmbeddings
from typing import List
import logging

def get_embeddings(model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
    """
    Creates and returns an OpenAIEmbeddings object.

    Args:
        model (str): The name of the OpenAI model to use for embeddings.

    Returns:
        OpenAIEmbeddings: An instance of OpenAIEmbeddings.
    """
    try:
        return OpenAIEmbeddings(model=model)
    except Exception as e:
        logging.error(f"Error creating OpenAIEmbeddings: {str(e)}")
        raise

def create_embeddings(texts: List[str], embeddings: OpenAIEmbeddings) -> List[List[float]]:
    """
    Creates embeddings for a list of texts.

    Args:
        texts (List[str]): A list of texts to create embeddings for.
        embeddings (OpenAIEmbeddings): An instance of OpenAIEmbeddings.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.
    """
    try:
        return embeddings.embed_documents(texts)
    except Exception as e:
        logging.error(f"Error creating embeddings: {str(e)}")
        raise
