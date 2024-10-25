import json
import copy
import uuid
import pandas as pd
import requests
import chardet
import re
from typing import Dict, Any, List, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import create_pandas_dataframe_agent, AgentType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class CSVRetriever:
    """Handles retrieval and processing of CSV documents."""
    
    def __init__(self, connection_string: str, collection_name: str, openai_api_key: str):
        """Initialize the CSV retriever with required connections."""
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.read_only_memory = ReadOnlySharedMemory(memory=self.memory)
        self.handler = StreamingStdOutCallbackHandler()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            streaming=True,
            callbacks=[self.handler],
            temperature=0,
            model="gpt-4-turbo",
            openai_api_key=openai_api_key
        )

    def _get_vector_store(self, metadata: Dict[str, Any]) -> PGVector:
        """Get vector store with embeddings."""
        return PGVector.from_existing_index(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection_string=self.connection_string
        )

    def _process_google_sheet_url(self, url: str) -> str:
        """Convert Google Sheet URL to CSV download URL if needed."""
        if 'docs.google.com/spreadsheets' in url:
            extracted_id = re.search(r'/d/(.+?)/edit', url)
            if extracted_id:
                return f"https://docs.google.com/spreadsheets/d/{extracted_id.group(1)}/gviz/tq?tqx=out:csv"
        return url

    def _download_csv(self, url: str) -> str:
        """Download CSV file and return local filename."""
        local_filename = f"/tmp/uuid-{uuid.uuid4()}"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def _create_pandas_agent(self, df: pd.DataFrame):
        """Create pandas dataframe agent."""
        return create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )

    def _get_response_format(self) -> Tuple[StructuredOutputParser, str]:
        """Create response format parser and instructions."""
        schemas = [
            ResponseSchema(name="final answer", description="the answer of the question"),
            ResponseSchema(name="source metadata", description="metadata of the source used"),
            ResponseSchema(name="further_questions", description="list of suggested questions")
        ]
        parser = StructuredOutputParser.from_response_schemas(schemas)
        return parser, parser.get_format_instructions()

    def process_query(self, question: str, metadata_input: Dict[str, Any]) -> str:
        """
        Process a question against CSV documents.
        
        Args:
            question: The query to process
            metadata_input: Metadata for filtering documents
        
        Returns:
            JSON string containing answer, metadata, and follow-up questions
        """
        # Prepare metadata
        csv_metadata = copy.deepcopy(metadata_input)
        csv_metadata['data_type'] = 'spreadsheet'

        # Get relevant document
        vector_store = self._get_vector_store(csv_metadata)
        question_embedding = self.embeddings.embed_query(question)
        relevant_docs = vector_store.max_marginal_relevance_search_with_score_by_vector(
            embedding=question_embedding,
            k=1,
            fetch_k=10,
            lambda_mult=0.5,
            filter=csv_metadata
        )

        if not relevant_docs:
            return json.dumps({"error": "No relevant documents found"})

        # Process most relevant document
        doc, score = relevant_docs[0]
        csv_url = doc.metadata["file_list"]
        if isinstance(csv_url, list):
            csv_url = csv_url[0]

        # Handle Google Sheets
        csv_url = self._process_google_sheet_url(csv_url)
        
        # Load CSV
        df = pd.read_csv(csv_url, sep=',', header=None)
        
        # Create pandas agent and get initial answer
        agent = self._create_pandas_agent(df)
        initial_answer = agent.run(question)

        # Format final response
        parser, format_instructions = self._get_response_format()
        
        # Create prompt for detailed answer
        answer_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "Using the question: {question}, initial answer: {initial_answer}, "
                    "and document metadata: {doc_info}, provide a comprehensive response."
                )
            ],
            input_variables=["question", "initial_answer", "doc_info"],
            partial_variables={"format_instructions": format_instructions},
            output_parser=parser
        )

        # Get final answer
        chain = LLMChain(llm=self.llm, prompt=answer_prompt)
        result = chain.invoke({
            "question": question,
            "initial_answer": initial_answer,
            "doc_info": f"{doc.page_content}, {doc.metadata}"
        })

        # Get follow-up questions
        questions_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "Based on the question and answer, suggest three specific follow-up questions."
                )
            ],
            input_variables=[],
            partial_variables={"format_instructions": format_instructions},
            output_parser=parser
        )
        
        questions_chain = LLMChain(llm=self.llm, prompt=questions_prompt)
        further_questions = questions_chain.invoke({})["text"].split('\n')
        further_questions = [q for q in further_questions if q.strip()]

        # Prepare final response
        response = {
            "final_answer": result["text"],
            "source_metadatas": [doc.metadata],
            "further_questions": further_questions
        }

        return json.dumps(response)