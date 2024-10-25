from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from core.file_summary import DocumentSummaries
from typing import List
from langchain.schema import Document

def load_worddoc(file_path: str, file_type: str, metadata: dict, connection_string: str) -> None:
    """
    Load and process Word documents, create vector store representations, and generate summaries.
    
    Args:
        file_path: Path to the Word document
        file_type: Type of file (doc/docx)
        metadata: Metadata to attach to documents
        connection_string: Database connection string for vector store
    """
    print(f"Processing Word document: {file_path}")
    
    try:
        # Initialize components
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()
        metadata["file_type"] = "doc"

        # Load document
        loader = Docx2txtLoader(file_path)
        full_document = loader.load()

        if not full_document:
            print("Warning: Empty document loaded")
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

        # Create vector stores
        vector_store = PGVector.from_documents(
            documents=split_documents,
            embedding=embeddings,
            connection_string=connection_string
        )
        
        full_doc_vector_store = PGVector.from_documents(
            documents=full_document,
            embedding=embeddings,
            ids=["full doc"],
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
        
        print("Document processing complete")
        
    except Exception as e:
        print(f"Error processing Word document: {str(e)}")
        raise