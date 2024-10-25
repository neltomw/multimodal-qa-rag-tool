
'''from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_googledrive.document_loaders import GoogleDriveLoader
from langchain.document_loaders import UnstructuredFileIOLoader
from core.file_summary import DocumentSummaries
import re
from typing import Tuple, List, Optional
from langchain.schema import Document

def extract_google_id(file_url: str) -> Optional[str]:
    """Extract Google Doc/Sheet ID from URL."""
    match = re.search(r'/d/(.+?)/edit', file_url)
    return match.group(1) if match else None

def load_googledoc(file_path: str, file_type: str, metadata: dict, connection_string: str) -> None:
    """
    Load and process Google Docs/Sheets, create vector store representations, and generate summaries.
    
    Args:
        file_path: URL of the Google document
        file_type: Type of file ('googledoc' or 'googleexcel')
        metadata: Metadata to attach to documents
        connection_string: Database connection string for vector store
    """
    print(f"Processing Google document: {file_path}")
    
    try:
        # Initialize components
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()
        document_id = extract_google_id(file_path)
        
        if not document_id:
            raise ValueError("Could not extract Google document ID from URL")

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=0,
            separators=[" ", ",", "\n"]
        )

        if file_type == 'googledoc':
            # Load Google Doc
            loader = GoogleDriveLoader(
                document_ids=[document_id],
                num_results=1,
                supportsAllDrives=False,
            )
            
            full_document = loader.load()
            if not full_document:
                print("Warning: Empty document loaded")
                return
                
            # Split document into chunks
            split_documents = text_splitter.split_documents(full_document)
            
            # Add metadata
            for doc in split_documents:
                doc.metadata = metadata
                
            # Generate document summary
            doc_summary = loader.lazy_update_description_with_summary(llm=llm)
            print(f"Document summary: {doc_summary}")

        elif file_type == 'googleexcel':
            # Load Google Sheet
            loader = GoogleDriveLoader(
                folder_id=document_id,
                template="gdrive-by-name",
                file_loader_cls=UnstructuredFileIOLoader,
                file_loader_kwargs={"mode": "elements"}
            )
            
            full_document = loader.load()
            if not full_document:
                print("Warning: Empty spreadsheet loaded")
                return
                
            # Split document into chunks
            split_documents = text_splitter.split_documents(full_document)

        else:
            raise ValueError(f"Unsupported Google document type: {file_type}")

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
        
        print(f"Google {file_type} processing complete")
        
    except Exception as e:
        print(f"Error processing Google document: {str(e)}")
        raise
        '''