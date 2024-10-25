from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from core.file_summary import DocumentSummaries
from typing import List
from langchain.schema import Document

def load_txt(file_path: str, file_type: str, metadata: dict, connection_string: str) -> None:
    """
    Load and process text files, create vector store representations, and generate summaries.
    
    Args:
        file_path: Path to the text file
        file_type: Type of file ('txt')
        metadata: Metadata to attach to documents
        connection_string: Database connection string for vector store
    """
    print(f"Processing text file: {file_path}")
    
    try:
        # Initialize components
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()
        metadata["file_type"] = "txt"

        # Load text file
        loader = TextLoader(file_path)
        full_document = loader.load()
        
        if not full_document:
            print("Warning: Empty text file loaded")
            return

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_documents = text_splitter.split_documents(full_document)

        # Add metadata to all document chunks
        for doc in split_documents:
            doc.metadata = metadata

        # Create vector store
        vector_store = PGVector.from_documents(
            documents=split_documents,
            embedding=embeddings,
            connection_string=connection_string
        )

        # Generate document summaries
        DocumentSummaries(
            file_path=file_path,
            file_type=file_type,
            full_doc=full_document,
            docs=split_documents,
            metadata=metadata,
            connection_string=connection_string
        )
        
        print("Text file processing complete")
        
    except Exception as e:
        print(f"Error processing text file: {str(e)}")
        raise