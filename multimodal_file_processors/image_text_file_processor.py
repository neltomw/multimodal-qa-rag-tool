import torch
from PIL import Image
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Union, Optional
from transformers import CLIPProcessor, CLIPModel
from langchain.vectorstores.pgvector import PGVector
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

class CLIPMultimodalProcessor:
    """Handles multimodal data processing using CLIP model."""
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 connection_string: Optional[str] = None):
        """Initialize CLIP model and processor."""
        print("CLIPMultimodalProcessor init")
        print("model_name", model_name)
        print("connection_string", connection_string)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.connection_string = connection_string

    def process_image(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """Process image and return CLIP embeddings."""
        if isinstance(image, str):
            image = Image.open(image)
            
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return self._normalize_features(image_features)

    def process_text(self, text: str) -> torch.Tensor:
        """Process text and return CLIP embeddings."""
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        return self._normalize_features(text_features)

    def calculate_similarity_matrix(self, 
                                  image_features: List[torch.Tensor], 
                                  text_features: List[torch.Tensor]) -> np.ndarray:
        """Calculate similarity matrix between all images and texts."""
        similarity_matrix = np.zeros((len(image_features), len(text_features)))
        for i, img_feat in enumerate(image_features):
            for j, txt_feat in enumerate(text_features):
                similarity_matrix[i, j] = self.calculate_similarity(img_feat, txt_feat)
        return similarity_matrix

    def calculate_similarity(self, 
                           image_features: torch.Tensor, 
                           text_features: torch.Tensor) -> float:
        """Calculate cosine similarity between image and text features."""
        return torch.matmul(text_features, image_features.T).item()

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize feature vectors."""
        return features / features.norm(dim=-1, keepdim=True)

class DataVisualizer:
    """Handles visualization of similarity scores and embeddings."""
    
    @staticmethod
    def create_heatmap(similarity_matrix: np.ndarray, 
                      image_labels: List[str], 
                      text_labels: List[str]) -> go.Figure:
        """Create interactive heatmap using plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=text_labels,
            y=image_labels,
            colorscale='Viridis',
            hoverongaps=False))
        
        fig.update_layout(
            title='Image-Text Similarity Heatmap',
            xaxis_title='Text Descriptions',
            yaxis_title='Images',
            height=400
        )
        return fig

    @staticmethod
    def create_similarity_bar_chart(similarities: List[float], 
                                  labels: List[str]) -> go.Figure:
        """Create interactive bar chart for similarities."""
        fig = go.Figure(data=go.Bar(
            x=labels,
            y=similarities,
            marker_color=similarities,
            marker_colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Similarity Scores',
            xaxis_title='Pairs',
            yaxis_title='Similarity Score',
            height=400
        )
        return fig

    @staticmethod
    def display_image_grid(images: List[Image.Image], 
                          cols: int = 3) -> None:
        """Display images in a grid layout."""
        rows = len(images) // cols + (1 if len(images) % cols else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        for idx, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Image {idx+1}')
            
        # Hide empty subplots
        for idx in range(len(images), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def create_streamlit_ui():
    """Create enhanced Streamlit UI with visualizations."""
    st.set_page_config(layout="wide")
    st.title("Image-Text Similarity Analysis")
    st.write("Upload images and enter text descriptions to analyze similarities.")

    # Initialize processors
    print("CLIPMultimodalProcessor A")
    clip_processor = CLIPMultimodalProcessor()
    print("CLIPMultimodalProcessor B")
    visualizer = DataVisualizer()

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        visualization_type = st.selectbox(
            "Choose Visualization",
            ["Heatmap", "Bar Chart", "Both"]
        )
        show_details = st.checkbox("Show Detailed Scores", value=True)
        
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Images")
        uploaded_images = st.file_uploader(
            "Choose images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_images:
            images = [Image.open(img) for img in uploaded_images]
            st.subheader("Uploaded Images")
            visualizer.display_image_grid(images)

    with col2:
        st.subheader("Enter Text Descriptions")
        text_input = st.text_area(
            "Enter descriptions (one per line)",
            height=200
        )
        texts = [t.strip() for t in text_input.split('\n') if t.strip()]

    if uploaded_images and texts:
        st.header("Analysis Results")

        try:
            print("CLIPMultimodalProcessor C")
            # Process images and texts
            image_embeddings = [clip_processor.process_image(img) for img in images]
            text_embeddings = [clip_processor.process_text(text) for text in texts]
            
            # Calculate similarity matrix
            similarity_matrix = clip_processor.calculate_similarity_matrix(
                image_embeddings, 
                text_embeddings
            )
            print("CLIPMultimodalProcessor D")

            # Create visualizations based on selection
            if visualization_type in ["Heatmap", "Both"]:
                st.subheader("Similarity Heatmap")
                image_labels = [f"Image {i+1}" for i in range(len(images))]
                text_labels = [f"Text {i+1}" for i in range(len(texts))]
                heatmap = visualizer.create_heatmap(
                    similarity_matrix,
                    image_labels,
                    text_labels
                )
                st.plotly_chart(heatmap, use_container_width=True)

            if visualization_type in ["Bar Chart", "Both"]:
                st.subheader("Similarity Scores")
                # Flatten similarity matrix for bar chart
                similarities = similarity_matrix.flatten()
                labels = [
                    f"Img{i+1}-Txt{j+1}" 
                    for i in range(len(images)) 
                    for j in range(len(texts))
                ]
                bar_chart = visualizer.create_similarity_bar_chart(
                    similarities.tolist(),
                    labels
                )
                st.plotly_chart(bar_chart, use_container_width=True)

            # Show detailed scores if requested
            if show_details:
                st.subheader("Detailed Similarity Scores")
                for i in range(len(images)):
                    expander = st.expander(f"Image {i+1} Similarities")
                    with expander:
                        for j in range(len(texts)):
                            score = similarity_matrix[i, j]
                            st.write(f"Text {j+1}: {score:.4f}")
                            st.write(f"Description: {texts[j]}")
                            st.write("---")

        except Exception as e:
            st.error(f"Error processing inputs: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    create_streamlit_ui()