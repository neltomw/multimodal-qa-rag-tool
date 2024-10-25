from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from core.file_summary import DocumentSummaries
from typing import Dict, Any, List, Optional
from langchain.schema import Document
import logging

class load_youtube:
    """Handles loading and processing of YouTube video content."""
    
    def __init__(self, connection_string: str):
        """
        Initialize YouTube content processor.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        
        # Initialize core components
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        self.embeddings = OpenAIEmbeddings()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.read_only_memory = ReadOnlySharedMemory(memory=self.memory)
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def process_video(self, video_url: str, file_type: str) -> None:
        """
        Process YouTube video content.
        
        Args:
            video_url: URL of the YouTube video
            file_type: Type of file being processed
        """
        logging.info(f"Processing YouTube video: {video_url}")
        
        try:
            # Load video content
            video_content = self._load_video_content(video_url)
            if not video_content:
                raise ValueError("No content loaded from video")

            # Log video metadata
            self._log_video_info(video_content[0].metadata)
            
            # Split content into chunks
            split_documents = self._split_content(video_content)
            
            # Create vector store
            vector_store = self._create_vector_store(split_documents)
            
            # Process with document summaries
            self._process_summaries(
                video_url,
                file_type,
                video_content,
                split_documents,
                split_documents[0].metadata
            )
            
            logging.info("Video processing complete")
            
        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            raise

    def _load_video_content(self, url: str) -> List[Document]:
        """Load content from YouTube video."""
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        return loader.load()

    def _split_content(self, content: List[Document]) -> List[Document]:
        """Split content into manageable chunks."""
        return self.text_splitter.split_documents(content)

    def _create_vector_store(self, documents: List[Document]) -> PGVector:
        """Create vector store from documents."""
        return PGVector.from_documents(
            documents=documents,
            embedding=self.embeddings,
            connection_string=self.connection_string
        )

    def _log_video_info(self, metadata: Dict[str, Any]) -> None:
        """Log video metadata information."""
        logging.info(
            f"Video by {metadata.get('author', 'Unknown')} - "
            f"Length: {metadata.get('length', 0)} seconds"
        )

    def _process_summaries(self,
                         file_path: str,
                         file_type: str,
                         full_content: List[Document],
                         split_content: List[Document],
                         metadata: Dict[str, Any]) -> None:
        """Process document summaries."""
        DocumentSummaries(
            file_path=file_path,
            file_type=file_type,
            full_doc=full_content,
            docs=split_content,
            metadata=metadata,
            connection_string=self.connection_string
        )

def create_youtube_processor(connection_string: str) -> load_youtube:
    """Factory function to create YouTubeProcessor instance."""
    return load_youtube(connection_string)
