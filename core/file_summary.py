import copy
import uuid
from typing import List, Tuple, Dict
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_core.output_parsers import StrOutputParser
from utils.embeddings import get_embeddings
from utils.vector_store import initialize_vector_store


class DocumentSummaries:
    def __init__(self, file_path: str, file_type: str, full_doc: List[Document], 
                 docs: List[Document], metadata: Dict, connection_string: str):
        self.connection_string = connection_string
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        self.embeddings = get_embeddings()
        self.vector_store = initialize_vector_store(self.connection_string, self.embeddings)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.read_only_memory = ReadOnlySharedMemory(memory=self.memory)

        self.summarize_file(file_path, file_type, docs, full_doc, metadata)

    def split_docs_into_nice_sizes(self, full_doc: List[Document], 
                                 input_metadata: Dict) -> List[Document]:
        """Split documents into manageable chunks."""
        combined_text = "".join([doc.page_content + "\n" for doc in full_doc])
        combined_doc = Document(page_content=combined_text, metadata=input_metadata)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=150000,
            length_function=len
        )
        split_docs = splitter.create_documents([combined_text])
        for doc in split_docs:
            doc.metadata = input_metadata
        return split_docs

    def process_documents(self, full_doc: List[Document], input_metadata: Dict, 
                        embeddings: OpenAIEmbeddings) -> Tuple[List[Document], List[Document]]:
        """Process documents to generate summaries and store in vector database."""
        # Create metadata for summaries
        full_doc_summary_metadata = copy.deepcopy(input_metadata)
        full_doc_summary_metadata['id_key'] = 'full_doc_summary'
        
        # Generate full document summary
        full_doc_chain = (
            {"full_doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document in four sentences:\n\n{full_doc}")
            | self.llm
            | StrOutputParser()
        )
        full_doc_summary = full_doc_chain.batch(full_doc, {"max_concurrency": 200})

        # Store vector representations
        vectorstore = PGVector.from_documents(
            documents=full_doc,
            embedding=embeddings,
            connection_string=self.connection_string
        )
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=InMemoryByteStore(),
            id_key="doc_id",
        )

        # Add summaries to vector store
        full_doc_ids = [str(uuid.uuid4()) for _ in full_doc]
        full_doc_summary_docs = [
            Document(page_content=s, metadata=full_doc_summary_metadata) 
            for s in full_doc_summary
        ]
        retriever.vectorstore.add_documents(full_doc_summary_docs)
        retriever.docstore.mset(list(zip(full_doc_ids, full_doc)))

        # Create comprehensive summary
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300000,
            length_function=len
        )
        complete_string = "\n".join([doc.page_content for doc in full_doc_summary_docs])

        comp_chain = (
            {"string_texts": lambda x: [complete_string]}
            | ChatPromptTemplate.from_template("Create a concise summary using five sentences:\n\n{string_texts}")
            | self.llm
            | StrOutputParser()
        )
        comp_summaries = comp_chain.batch([complete_string], {"max_concurrency": 1})

        # Store comprehensive summary
        comp_doc_summary_metadata = copy.deepcopy(input_metadata)
        comp_doc_summary_metadata['id_key'] = 'comp_doc_summary'

        comp_summaries_embedding = embeddings.embed_documents(comp_summaries)
        comp_vectorstore = PGVector.from_embeddings(
            text_embeddings=[[comp_summaries[0], comp_summaries_embedding[0]]], 
            embedding=embeddings, 
            metadatas=[comp_doc_summary_metadata], 
            connection_string=self.connection_string
        )
        final_comp_doc = text_splitter.create_documents(
            comp_summaries,
            metadatas=[comp_doc_summary_metadata]
        )

        return full_doc_summary_docs, final_comp_doc

    def summarize_file(self, file_path: str, file_type: str, docs: List[Document], 
                      full_doc: List[Document], metadata: Dict) -> Tuple[List[Document], List[Document]]:
        """Generate summaries for a file."""
        input_metadata = metadata
        full_doc = self.split_docs_into_nice_sizes(full_doc, input_metadata)
        
        # Use same processing for all file types
        embeddings = self.embeddings if file_type not in ['csv', 'xlsx'] else OpenAIEmbeddings()
        return self.process_documents(full_doc, input_metadata, embeddings)