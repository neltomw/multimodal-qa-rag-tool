from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from typing import Dict, Any
import json
import copy
import logging

class SummaryRetriever:
    """Handles retrieval and processing of document summaries."""
    
    def __init__(self, connection_string: str, openai_api_key: str):
        """
        Initialize the summary retriever.
        
        Args:
            connection_string: Database connection string
            openai_api_key: OpenAI API key for models
        """
        self.connection_string = connection_string
        self.openai_api_key = openai_api_key
        
        # Initialize core components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            streaming=True,
            temperature=0,
            model="gpt-4-turbo",
            openai_api_key=openai_api_key
        )
        
        # Initialize vector store
        self.vector_store = PGVector.from_existing_index(
            embedding=self.embeddings,
            connection_string=self.connection_string
        )

    def retrieve_summary(self, question: str, metadata: Dict[str, Any]) -> str:
        """
        Retrieve and process document summaries based on a question.
        
        Args:
            question: Question to process
            metadata: Metadata for filtering documents
            
        Returns:
            JSON string containing answer, metadata, and follow-up questions
        """
        try:
            # Prepare metadata filters
            summary_metadata = self._prepare_metadata_filters(metadata)
            
            # Create retriever with filters
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    'k': 1,
                    'filter': summary_metadata
                }
            )
            
            # Get relevant documents
            retrieved_docs = retriever.get_relevant_documents(query=question)
            
            if not retrieved_docs:
                return self._create_error_response("No relevant summaries found")
            
            # Set up QA chain for follow-up questions
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(
                qa_chain,
                question,
                retrieved_docs[0].page_content
            )
            
            # Format response
            response = {
                "final_answer": retrieved_docs[0].page_content,
                "source_metadatas": [retrieved_docs[0].metadata],
                "further_questions": follow_up_questions
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logging.error(f"Error retrieving summary: {str(e)}")
            return self._create_error_response(str(e))

    def _prepare_metadata_filters(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata filters for document retrieval."""
        return {
            'id_key': 'comp_doc_summary'
        }

    def _generate_follow_up_questions(self, 
                                    qa_chain: RetrievalQA, 
                                    question: str, 
                                    context: str) -> list:
        """Generate follow-up questions based on the context."""
        prompt = (
            f"The user asked the following question: '{question}'. "
            f"Given this context: '{context}', please provide 3 specific, "
            "further questions that can be answered based off the question, "
            "answer and context of the document. Do not number or add bullet "
            "points to the 3 questions, return them separated by new lines."
        )
        
        results = qa_chain(prompt)
        questions = results['result'].split('\n')
        
        # Clean up empty questions
        return [q for q in questions if q.strip()]

    def _create_error_response(self, error_message: str) -> str:
        """Create a standardized error response."""
        return json.dumps({
            "error": error_message,
            "final_answer": None,
            "source_metadatas": [],
            "further_questions": []
        })

def create_summary_retriever(connection_string: str, 
                           openai_api_key: str) -> SummaryRetriever:
    """Factory function to create a SummaryRetriever instance."""
    return SummaryRetriever(connection_string, openai_api_key)