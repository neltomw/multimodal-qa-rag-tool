
import torch
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from typing import List, Dict, Union, Optional, Tuple
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    WhisperProcessor, 
    WhisperForConditionalGeneration
)
import cv2
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import plotly.express as px
import io

class MultimediaProcessor:
    """Handles audio/video processing using Whisper and CLIP."""
    
    def __init__(self, 
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 whisper_model_name: str = "openai/whisper-base"):
        """Initialize models and processors."""
        # CLIP for visual processing
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Whisper for audio processing
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        self.whisper_model.to(self.device)

    def process_video(self, video_path: str, 
                     frame_interval: int = 30) -> Dict[str, Union[List[Image.Image], str]]:
        """
        Process video file: extract frames and transcribe audio.
        
        Args:
            video_path: Path to video file
            frame_interval: Number of frames to skip between extractions
        
        Returns:
            Dictionary containing frames and transcription
        """
        # Extract video frames
        frames = self._extract_video_frames(video_path, frame_interval)
        
        # Extract and transcribe audio
        audio = self._extract_audio(video_path)
        transcription = self.transcribe_audio(audio)
        
        return {
            "frames": frames,
            "transcription": transcription,
            "frame_embeddings": self._get_frame_embeddings(frames)
        }

    def process_audio(self, audio_path: str) -> Dict[str, Union[np.ndarray, str]]:
        """
        Process audio file: transcribe and analyze.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary containing audio features and transcription
        """
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path)
        
        # Get transcription
        transcription = self.transcribe_audio(audio)
        
        # Extract audio features
        features = self._extract_audio_features(audio, sr)
        
        return {
            "transcription": transcription,
            "features": features
        }

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper."""
        input_features = self.whisper_processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription

    def _extract_video_frames(self, video_path: str, 
                            frame_interval: int) -> List[Image.Image]:
        """Extract frames from video at specified intervals."""
        frames = []
        video = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                
            frame_count += 1
            
        video.release()
        return frames

    def _extract_audio(self, video_path: str) -> np.ndarray:
        """Extract audio from video file."""
        with VideoFileClip(video_path) as video:
            audio = video.audio
            # Export audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                audio.write_audiofile(temp_audio.name)
                # Load audio with librosa
                audio_data, _ = librosa.load(temp_audio.name, sr=16000)
        return audio_data

    def _get_frame_embeddings(self, frames: List[Image.Image]) -> torch.Tensor:
        """Get CLIP embeddings for video frames."""
        embeddings = []
        for frame in frames:
            inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
                embeddings.append(embedding)
        return torch.cat(embeddings)

    def _extract_audio_features(self, audio: np.ndarray, 
                              sr: int) -> Dict[str, np.ndarray]:
        """Extract audio features using librosa."""
        return {
            "mfcc": librosa.feature.mfcc(y=audio, sr=sr),
            "spectral_centroid": librosa.feature.spectral_centroid(y=audio, sr=sr),
            "chroma": librosa.feature.chroma_stft(y=audio, sr=sr)
        }

class MultimediaVisualizer:
    """Handles visualization of multimedia analysis results."""
    
    @staticmethod
    def display_video_frames(frames: List[Image.Image], cols: int = 4) -> None:
        """Display extracted video frames in a grid."""
        rows = len(frames) // cols + (1 if len(frames) % cols else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        axes = axes.flatten()
        
        for idx, (frame, ax) in enumerate(zip(frames, axes)):
            ax.imshow(frame)
            ax.axis('off')
            ax.set_title(f'Frame {idx+1}')
        
        # Hide empty subplots
        for idx in range(len(frames), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    @staticmethod
    def plot_audio_features(features: Dict[str, np.ndarray]) -> None:
        """Create interactive plots of audio features."""
        for feature_name, feature_data in features.items():
            fig = px.imshow(
                feature_data,
                aspect='auto',
                title=f'{feature_name.upper()} Visualization'
            )
            st.plotly_chart(fig)

def create_streamlit_ui():
    """Create Streamlit UI for multimedia processing."""
    st.set_page_config(layout="wide")
    st.title("Multimedia Analysis with Whisper and CLIP")
    
    # Initialize processors
    processor = MultimediaProcessor()
    visualizer = MultimediaVisualizer()
    
    # File upload
    file_type = st.radio("Select file type:", ["Video", "Audio"])
    uploaded_file = st.file_uploader(
        "Upload file", 
        type=['mp4', 'avi', 'mov', 'mp3', 'wav'] if file_type == "Video" 
        else ['mp3', 'wav']
    )
    
    if uploaded_file:
        with st.spinner("Processing file..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name
            
            try:
                if file_type == "Video":
                    # Process video
                    results = processor.process_video(file_path)
                    
                    # Display results
                    st.header("Video Analysis Results")
                    
                    # Display frames
                    st.subheader("Extracted Frames")
                    visualizer.display_video_frames(results["frames"])
                    
                    # Display transcription
                    st.subheader("Transcription")
                    st.write(results["transcription"])
                    
                else:
                    # Process audio
                    results = processor.process_audio(file_path)
                    
                    # Display results
                    st.header("Audio Analysis Results")
                    
                    # Display transcription
                    st.subheader("Transcription")
                    st.write(results["transcription"])
                    
                    # Display audio features
                    st.subheader("Audio Features")
                    visualizer.plot_audio_features(results["features"])
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)
            
            finally:
                # Cleanup
                Path(file_path).unlink()

if __name__ == "__main__":
    create_streamlit_ui()