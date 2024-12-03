import argparse
import logging
from typing import List, Dict, Any
from core.document_processor import DocumentProcessor
from operations.document_qa import QuestionHandler
from config import get_pgvector_connection_string

class RAGSystemCLI:
    """Command Line Interface for the RAG (Retrieval-Augmented Generation) System."""
    
    def __init__(self):
        """Initialize the RAG System CLI."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # System configuration
        self.connection_string = get_pgvector_connection_string()
        self.collection_id = '1234'
        self.default_metadata = {"source": "main_processor"}
        
        # Initialize processors
        self.doc_processor = DocumentProcessor(
            self.connection_string,
            self.collection_id,
            self.default_metadata
        )

    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """Set up command line argument parser."""
        parser = argparse.ArgumentParser(
            description="Document Processing and QA System",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        subparsers = parser.add_subparsers(
            dest="mode",
            required=True,
            help="Operation mode: process, ask, or both"
        )
        
        # Process mode parser
        process_parser = subparsers.add_parser(
            "process",
            help="Process documents"
        )
        self._add_file_arguments(process_parser)
        
        # Ask mode parser
        ask_parser = subparsers.add_parser(
            "ask",
            help="Ask questions about processed documents"
        )
        self._add_question_arguments(ask_parser)
        
        # Both mode parser
        both_parser = subparsers.add_parser(
            "both",
            help="Process documents and ask questions"
        )
        self._add_file_arguments(both_parser)
        self._add_question_arguments(both_parser)
        
        return parser

    def _add_file_arguments(self, parser: argparse.ArgumentParser):
        """Add file-related arguments to parser."""
        parser.add_argument(
            "--files",
            nargs="+",
            required=True,
            help="Paths to documents for processing"
        )
        parser.add_argument(
            "--types",
            nargs="+",
            required=True,
            help="Document types (pdf, txt, csv, jpeg, jpg, etc.)"
        )
        parser.add_argument(
            "--metadata",
            nargs="+",
            required=False,
            help="Additional metadata for documents"
        )

    def _add_question_arguments(self, parser: argparse.ArgumentParser):
        """Add question-related arguments to parser."""
        parser.add_argument(
            "--questions",
            nargs="+",
            required=True,
            help="Questions to ask about the documents"
        )

    def process_documents(self, args: argparse.Namespace) -> List[Dict[str, Any]]:
        """Process documents based on provided arguments."""
        if len(args.files) != len(args.types):
            raise ValueError("Number of files and file types must match")
            
        logging.info(f"Processing {len(args.files)} documents...")
        results = self.doc_processor.process_file(
            args.files,
            args.types,
            self.connection_string,
            self.default_metadata
        )
        
        self._display_processing_results(results)
        return results

    def handle_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process questions and display results."""
        logging.info(f"Processing {len(questions)} questions...")
        
        qa_handler = QuestionHandler(questions, self.connection_string)
        results = qa_handler.process_questions(questions)
        
        self._display_qa_results(results)
        return results

    def _display_processing_results(self, results: List[Dict[str, Any]]):
        """Display document processing results."""
        for result in results:
            if result["status"] == "success":
                logging.info(f"✓ Successfully processed: {result['file_path']}")
            else:
                logging.error(f"✗ Failed to process: {result['file_path']}")
                logging.error(f"  Error: {result['error_message']}")

    def _display_qa_results(self, results: List[Dict[str, Any]]):
        """Display question-answering results."""
        for result in results:
            if "error" in result:
                logging.error(f"Error: {result['error']}")
                continue
                
            print("\n" + "="*50)
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            
            print("\nSource Documents:")
            for doc in result['source_documents']:
                print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")
            
            print("\nFollow-up Questions:")
            for q in result['follow_up_questions']:
                print(f"- {q}")

    def run(self):
        """Run the RAG System CLI."""
        parser = self.setup_argument_parser()
        args = parser.parse_args()
        
        logging.info(f"Running in {args.mode} mode")
        logging.info(f"Using database connection: {self.connection_string}")
        
        try:
            if args.mode in ["process", "both"]:
                self.process_documents(args)
                
            if args.mode in ["ask", "both"]:
                self.handle_questions(args.questions)
                
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            raise

def main():
    """Main entry point for the RAG System CLI."""
    cli = RAGSystemCLI()
    cli.run()

if __name__ == "__main__":
    main()
