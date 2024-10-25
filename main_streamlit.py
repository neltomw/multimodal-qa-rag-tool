import logging
import os
import streamlit as st
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any
from core.document_processor import DocumentProcessor
from operations.document_qa import QuestionHandler
from config import get_pgvector_connection_string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RAGSystemUI:
    """Handles the Streamlit UI for the RAG system."""
    
    def __init__(self):
        """Initialize the RAG System UI."""
        self.setup_page_config()
        self.connection_string = get_pgvector_connection_string()
        self.collection_id = '1234'
        self.supported_file_types = {
            "PDF": "pdf",
            "Text": "txt",
            "CSV": "csv",
            "Word": "docx",
            "PowerPoint": "pptx",
            "Excel": "xlsx"
        }

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Multimodal RAG System",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add custom CSS
        st.markdown("""
            <style>
                .main {
                    padding: 2rem;
                }
                .stButton>button {
                    width: 100%;
                    margin-top: 1rem;
                }
                .success-message {
                    padding: 1rem;
                    background-color: #d4edda;
                    border-radius: 0.5rem;
                    margin: 1rem 0;
                }
                .error-message {
                    padding: 1rem;
                    background-color: #f8d7da;
                    border-radius: 0.5rem;
                    margin: 1rem 0;
                }
            </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Mode selection
            mode = st.radio(
                "Operation Mode",
                ["Process Files 📄", "Ask Questions ❓", "Process and Ask 🔄"],
                help="Select how you want to use the system"
            )
            
            # Advanced settings expandable section
            with st.expander("🛠️ Advanced Settings"):
                connection_string = st.text_input(
                    "Vector DB Connection",
                    value=self.connection_string,
                    type="password",
                    help="Database connection string"
                )
                
                metadata = st.text_area(
                    "Metadata (JSON)",
                    value="{}",
                    help="Additional metadata in JSON format"
                )
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")
                    metadata = {}
            
            return {
                "mode": mode,
                "connection_string": connection_string,
                "metadata": metadata
            }

    def process_files(self, connection_string: str, metadata: Dict[str, Any]):
        """Handle file processing interface."""
        st.header("📁 File Processing")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Upload your documents",
                type=list(self.supported_file_types.values()),
                accept_multiple_files=True,
                help="Select one or more files to process"
            )
        
        with col2:
            if uploaded_files:
                st.info(f"📊 {len(uploaded_files)} files selected")
                file_types = []
                for file in uploaded_files:
                    suffix = Path(file.name).suffix[1:].lower()
                    file_types.append(suffix)
                    st.success(f"✓ Detected: {file.name} ({suffix})")

        if uploaded_files:
            if st.button("🚀 Process Files", use_container_width=True):
                processor = DocumentProcessor(connection_string, self.collection_id, metadata)
                
                with st.spinner("Processing files..."):
                    progress_bar = st.progress(0)
                    for idx, (file, file_type) in enumerate(zip(uploaded_files, file_types)):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                            tmp_file.write(file.read())
                            tmp_path = tmp_file.name
                            
                        try:
                            results = processor.process_file(
                                [tmp_path], 
                                [file_type], 
                                connection_string, 
                                metadata
                            )
                            
                            for result in results:
                                if result["status"] == "success":
                                    st.success(f"✅ Successfully processed: {file.name}")
                                else:
                                    st.error(f"❌ Failed to process {file.name}: {result['error_message']}")
                                    
                        finally:
                            os.remove(tmp_path)
                            
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.balloons()

    def ask_questions(self, connection_string: str):
        """Handle question answering interface."""
        st.header("❓ Question Answering")
        
        # Question input
        question_input = st.text_area(
            "Enter your questions",
            placeholder="Enter one question per line...",
            help="You can enter multiple questions, one per line"
        )
        
        if question_input:
            questions = [q.strip() for q in question_input.split("\n") if q.strip()]
            
            if st.button("🔍 Get Answers", use_container_width=True):
                processor = QuestionHandler(questions, connection_string)
                
                with st.spinner("Analyzing questions..."):
                    results = processor.process_questions(questions)
                    
                    for result in results:
                        with st.container():
                            st.markdown("---")
                            st.markdown(f"### Question\n{result['question']}")
                            
                            if "error" in result:
                                st.error(f"❌ {result['error']}")
                            else:
                                st.markdown("### Answer")
                                st.write(result['answer'])
                                
                                with st.expander("📚 Source Documents"):
                                    for doc in result['source_documents']:
                                        st.markdown(
                                            f"**Source**: {doc.metadata['source']}\n\n"
                                            f"**Content**: {doc.page_content[:200]}..."
                                        )
                                
                                with st.expander("🤔 Follow-up Questions"):
                                    for q in result['follow_up_questions']:
                                        st.markdown(f"- {q}")

    def run(self):
        """Run the main application."""
        st.title("🤖 Multimodal RAG System")
        st.markdown("---")
        
        # Get configuration from sidebar
        config = self.render_sidebar()
        
        # Main content area
        if "Process Files" in config["mode"]:
            self.process_files(config["connection_string"], config["metadata"])
            
        if "Ask Questions" in config["mode"]:
            self.ask_questions(config["connection_string"])

def main():
    app = RAGSystemUI()
    app.run()

if __name__ == "__main__":
    main()