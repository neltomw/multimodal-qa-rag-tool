import uuid
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader, UnstructuredCSVLoader
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from core.file_summary import DocumentSummaries

def load_csv(file_path: str, file_type: str, metadata: dict, connection_string: str) -> None:
    """
    Load and process CSV/Excel files, create vector store representations, and generate summaries.
    
    Args:
        file_path: Path to the CSV/Excel file
        file_type: Type of file ('csv' or 'xlsx')
        metadata: Metadata to attach to documents
        connection_string: Database connection string for vector store
    """
    print("Starting CSV processing")
    
    # Initialize components
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    embeddings = OpenAIEmbeddings()
    metadata["data_type"] = "spreadsheet"
    
    # Handle Excel files by converting to CSV
    if file_type == "xlsx":
        print("Converting XLSX to CSV")
        df = pd.read_excel(file_path, engine='openpyxl')
        temp_csv_path = f"{file_path}_converted.csv"
        df.to_csv(temp_csv_path, index=False)
        file_path = temp_csv_path
        print("Conversion complete")
    
    # Load CSV content in two formats
    try:
        full_csv_loader = UnstructuredCSVLoader(file_path, mode="single")
        csv_loader = CSVLoader(file_path)
        
        # Load documents and add metadata
        full_document = full_csv_loader.load()
        csv_documents = csv_loader.load()
        for doc in csv_documents:
            doc.metadata = metadata
        print("Metadata added to documents")
        
        # Initialize memory components
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        read_only_memory = ReadOnlySharedMemory(memory=memory)
        
        # Create vector stores
        vector_store = PGVector.from_documents(
            documents=csv_documents,
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
            docs=csv_documents,
            metadata=metadata,
            connection_string=connection_string
        )
        
        print("File processing complete")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise