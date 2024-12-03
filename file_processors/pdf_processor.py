from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from core.file_summary import DocumentSummaries
from typing import List
from langchain.schema import Document

def load_pdf(file_path: str, file_type: str, metadata: dict, connection_string: str) -> None:
    """
    Load and process PDF documents, create vector store representations, and generate summaries.
    
    Args:
        file_path: Path to the PDF document
        file_type: Type of file ('pdf')
        metadata: Metadata to attach to documents
        connection_string: Database connection string for vector store
    """
    print(f"Processing PDF document: {file_path}")
    
    try:
        print(f"Processing A")
        # Initialize components
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()
        metadata["file_type"] = "pdf"

        print(f"Processing B")

        # Load PDF
        loader = PyPDFLoader(file_path)
        print(f"Processing C")
        full_document = loader.load()
        print(f"Processing D")

        if not full_document:
            print("Warning: Empty PDF document loaded")
            return

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        print(f"Processing E")
        split_documents = text_splitter.split_documents(full_document)
        print(f"Processing F")

        # Add metadata to all document chunks
        for doc in split_documents:
            doc.metadata = metadata

        print(f"Processing G")
        # Initialize memory components
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        read_only_memory = ReadOnlySharedMemory(memory=memory)

        print(f"Processing H")
        # Create vector stores
        vector_store = PGVector.from_documents(
            documents=split_documents,
            embedding=embeddings,
            connection_string=connection_string
        )
        
        print(f"Processing I")
        full_doc_vector_store = PGVector.from_documents(
            documents=full_document,
            embedding=embeddings,
            ids=["full doc"],
            connection_string=connection_string
        )
        print(f"Processing J")

        # Generate document summaries
        DocumentSummaries(
            file_path=file_path,
            file_type=file_type,
            full_doc=full_document,
            docs=split_documents,
            metadata=metadata,
            connection_string=connection_string
        )
        print(f"Processing K")
        
        print("PDF processing complete")
        
    except Exception as e:
        print(f"Error processing PDF document: {str(e)}")
        raise