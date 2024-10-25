from langchain.agents import Tool
import operations.document_qa
import operations.retrieve_summaries
import langchain

langchain.debug = True

def get_tools(processor, questions, connection_string=None, metadata=None):

    tools = [
        Tool(
            name="document summary tool",
            func=lambda q: operations.retrieve_summaries.retrieve_summary(questions, connection_string, metadata),
            description="Useful for summarizing documents. Input should be a request to summarize a document, presentation, discussion, class topic or class session.",
            return_direct=True,
        ),
        Tool(
            name="doc search QA system",
            func= lambda q: processor.process_questions([q]),
            description="Useful for when you need to answer specific questions about documents, not for generating summaries. Input should be a fully formed question.",
            return_direct=True,
        )
    ]
    return tools

''' 
        Tool(
            name="data querying QA system",
            func=retrieve_csv_doc,
            description="Useful for when you need to answer questions about factual data. This includes information contained within a csv, google spreadsheet or excel file. Input should be a fully formed question that is asking about facts.",
            return_direct=True,
        ),  
        Tool(
            name="document grading and evaluation tool",
            func=document_grading_evaluation,
            description="Useful for when you need to grade or evaluate a document.",
        ),
        Tool(
            name="study guide generator",
            func=study_guide_generator,
            description="Useful for when you need to create study material, such as a study guide.",
        )
    '''    