from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from typing import List, Dict, Any, Optional
import logging
import agents.tools
import agents.prompts
from agents.prompts import RunGetTools

class QuestionHandler:
    """Handles processing and answering questions using LLMs and agents."""
    
    def __init__(self, questions: List[str], connection_string: str):
        """
        Initialize the question handler.
        
        Args:
            questions: List of questions to process
            connection_string: Database connection string
        """
        logging.info(f"Initializing QuestionHandler with {len(questions)} questions")
        
        self.questions = questions
        self.connection_string = connection_string
        
        # Initialize components
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.read_only_memory = ReadOnlySharedMemory(memory=self.memory)
        
        # Get tools and prompts
        self.tools = agents.tools.get_tools(self, questions, connection_string)
        self.prompt = agents.prompts.RunGetTools.process(self, questions)

    def process_single_question(self, question: str) -> Dict[str, Any]:
        """
        Process a single question and return formatted response.
        
        Args:
            question: Question to process
            
        Returns:
            Dictionary containing answer, sources, and follow-up questions
        """
        logging.info(f"Processing question: {question}")
        
        try:
            # Create agent chain
            llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
            agent = ZeroShotAgent(
                llm_chain=llm_chain,
                tools=self.tools,
                verbose=True
            )
            
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )

            # Get response from agent
            response = agent_executor(question)
            return self._format_response(response, question)
            
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            return self._create_error_response(question, str(e))

    def process_questions(self) -> List[Dict[str, Any]]:
        """
        Process all questions in the queue.
        
        Returns:
            List of response dictionaries
        """
        logging.info(f"Processing {len(self.questions)} questions")
        return [self.process_single_question(question) for question in self.questions]

    def _format_response(self, response: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Format the agent's response into a standardized structure."""
        return {
            "question": question,
            "answer": response['output'],
            "source_documents": response.get('source_documents', []),
            "chat_history": self.memory.chat_memory.messages,
            "follow_up_questions": self._generate_follow_up_questions(response['output'])
        }

    def _create_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "question": question,
            "error": error,
            "answer": None,
            "source_documents": [],
            "chat_history": [],
            "follow_up_questions": []
        }

    def _generate_follow_up_questions(self, answer: str) -> List[str]:
        """Generate follow-up questions based on the answer."""
        try:
            prompt = (
                "Based on this answer, generate 3 relevant follow-up questions "
                f"that would help explore the topic further: '{answer}'"
            )
            response = self.llm(prompt)
            return [q.strip() for q in response.strip().split('\n') if q.strip()]
            
        except Exception as e:
            logging.error(f"Error generating follow-up questions: {str(e)}")
            return []

def create_question_handler(questions: List[str], 
                          connection_string: str) -> QuestionHandler:
    """Factory function to create a QuestionHandler instance."""
    return QuestionHandler(questions, connection_string)
