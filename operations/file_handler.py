import logging
from typing import List
from core.document_processor import DocumentProcessor

def process_files(processor: DocumentProcessor, file_paths: List[str], file_types: List[str]) -> List[dict]:
    results = []
    for file_path, file_type in zip(file_paths, file_types):
        try:
            processor.process_file(file_path, file_type)
            logging.info(f"Successfully processed {file_path}")
            results.append({
                "file_path": file_path,
                "file_type": file_type,
                "status": "success"
            })
        except Exception as e:
            error_message = f"Error processing {file_path}: {str(e)}"
            logging.error(error_message)
            results.append({
                "file_path": file_path,
                "file_type": file_type,
                "status": "error",
                "error_message": error_message
            })
    return results