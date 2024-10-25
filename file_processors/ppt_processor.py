from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from core.file_summary import DocumentSummaries
from typing import List
from langchain.schema import Document

def load_ppt(file_path: str, file_type: str, metadata: dict, connection_string: str) -> None:
    """
    Load and process PowerPoint documents, create vector store representations, and generate summaries.
    
    Args:
        file_path: Path to the PowerPoint document
        file_type: Type of file ('ppt' or 'pptx')
        metadata: Metadata to attach to documents
        connection_string: Database connection string for vector store
    """
    print("Starting PowerPoint processing")
    
    try:
        # Initialize components
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()
        metadata["file_type"] = "ppt"

        # Load full presentation content
        full_ppt_loader = UnstructuredPowerPointLoader(file_path, mode="single")
        full_document = full_ppt_loader.load()
        
        if not full_document:
            print("Warning: Empty PowerPoint document loaded")
            return

        # Add metadata to full document
        for slide in full_document:
            slide.metadata = metadata

        # Load presentation by elements (slides)
        element_loader = UnstructuredPowerPointLoader(file_path, mode="elements")
        split_documents = element_loader.load()

        # Add metadata to individual slides
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

        # Initialize memory components
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        read_only_memory = ReadOnlySharedMemory(memory=memory)

        # Generate document summaries
        DocumentSummaries(
            file_path=file_path,
            file_type=file_type,
            full_doc=full_document,
            docs=split_documents,
            metadata=metadata,
            connection_string=connection_string
        )
        
        print("PowerPoint processing complete")
        
    except Exception as e:
        print(f"Error processing PowerPoint document: {str(e)}")
        raise