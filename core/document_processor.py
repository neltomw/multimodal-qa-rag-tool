from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from typing import List, Dict, Any
import logging
from typing import List
from file_processors.pdf_processor import load_pdf
from file_processors.csv_processor import load_csv
from file_processors.docx_processor import load_worddoc
from file_processors.txt_processor import load_txt
from file_processors.ppt_processor import load_ppt
from web_file_processors.youtube_file_processor import load_youtube
from multimodal_file_processors.image_text_file_processor import CLIPMultimodalProcessor
#from google_file_processors.google_doc_processor import load_googledoc
#from google_file_processors.google_doc_processor import load_googledoc


class TXTProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
        self.metadata = metadata
    
    def process(self, file_path, file_type):
        return load_txt(file_path, file_type, self.metadata, self.connection_string)

class WordDocProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
        self.metadata = metadata
    
    def process(self, file_path, file_type):
        return load_worddoc(file_path, file_type, self.metadata, self.connection_string) 
    
class PDFProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
        self.metadata = metadata
    
    def process(self, file_path, file_type):
        return load_pdf(file_path, file_type, self.metadata, self.connection_string) 
    
class PPTProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
        self.metadata = metadata
    
    def process(self, file_path, file_type):
        return load_ppt(file_path, file_type, self.metadata, self.connection_string) 

class CSVProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
        self.metadata = metadata
    
    def process(self, file_path, file_type):
        return load_csv(file_path, file_type, self.metadata, self.connection_string)    

class YouTubeProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
    
    def process(self, file_path, file_type):
        return load_youtube(file_path, file_type, self.connection_string)

'''
class GoogleDocProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
    
    def process(self, file_path, file_type):
        return load_googledoc(file_path, file_type, self.connection_string)
'''
class ImageDocProcessor:
    def __init__(self, connection_string, metadata):
        self.connection_string = connection_string
    
    def process(self, file_path, file_type):
        print("CLIPMultimodalProcessor E")
        clip_processor = CLIPMultimodalProcessor(connection_string=self.connection_string)
        return clip_processor.process_image(file_path)
        
class DocumentProcessor:
    def __init__(self, connection_string: str, collection_ids: str, metadata: dict):
    
        self.connection_string = connection_string
        self.collection_ids = collection_ids
        self.metadata = metadata
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.read_only_memory = ReadOnlySharedMemory(memory=self.memory)

    def process_file(self, file_paths: list, file_types: list, connection_string: str, metadata:dict) -> list:
        results = []
    
        for file_path, file_type in zip(file_paths, file_types):
            try:
                processor_map = {
                    "pdf": PDFProcessor(connection_string, metadata),
                    "ppt": PPTProcessor(connection_string, metadata),
                    "csv": CSVProcessor(connection_string, metadata),
                    "xlsx": CSVProcessor(connection_string, metadata),
                    "docx": WordDocProcessor(connection_string, metadata),
                    "txt": TXTProcessor(connection_string, metadata),
                    "youtube": YouTubeProcessor(connection_string, metadata),
                    #"gdoc": GoogleDocProcessor(connection_string, metadata),
                    #"gexcel": GoogleDocProcessor(connection_string, metadata),
                    "image": ImageDocProcessor(connection_string, metadata),
                    "jpg": ImageDocProcessor(connection_string, metadata),
                    "jpeg": ImageDocProcessor(connection_string, metadata),
                    "png": ImageDocProcessor(connection_string, metadata)

                    }
                processor = processor_map.get(file_type)
                if processor:
                    processor_result = processor.process(file_path, file_type)
                    results.append({
                        "file_path": file_path,
                        "file_type": file_type,
                        "status": "success",
                        "result": processor_result
                    })
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
            except Exception as e:
                error_message = f"Error processing {file_path}: {str(e)}"
                logging.error(error_message)
                results.append({
                    "file_path": file_path,
                    "file_type": file_type,
                    "status": "error",
                    "error_message": error_message
                })
        print(results, "RESULTS")
        return results
    
    def ask_question(self, question: str) -> dict:
        try:
            llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )

            response = agent_chain(question)
            return self._format_response(response, question)
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "answer": None,
                "source_documents": [],
                "chat_history": [],
                "follow_up_questions": []
            }

    def ask_questions(self, questions: List[str]) -> list[dict]:
        return [self.ask_question(question) for question in questions]

    def _format_response(self, response: dict, question: str) -> dict:
        return {
            "question": question,
            "answer": response['output'],
            "source_documents": response.get('source_documents', []),
            "chat_history": self.memory.chat_memory.messages,
            "follow_up_questions": self._generate_follow_up_questions(response['output'])
        }

    def _generate_follow_up_questions(self, answer: str) -> List[str]:
        follow_up_prompt = f"Based on the answer: '{answer}', generate 3 follow-up questions."
        follow_up_response = self.llm(follow_up_prompt)
        #If LLM returns a newline-separated list of questions
        return follow_up_response.strip().split('\n')