import logging
from typing import List, Dict
from core.document_processor import DocumentProcessor

def ask_questions(processor: DocumentProcessor, questions: List[str]) -> List[Dict]:
    results = []
    for question in questions:
        try:
            result = processor.ask_question(question)
            results.append({
                "question": question,
                "answer": result['answer'],
                "source_documents": result['source_documents']
            })
        except Exception as e:
            logging.error(f"Error answering question '{question}': {str(e)}")
            results.append({
                "question": question,
                "error": str(e)
            })
    return results