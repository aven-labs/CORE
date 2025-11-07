"""
Speech-to-text utilities using Azure Foundry Whisper
"""
import speech_recognition as sr
import requests
import io
import tempfile
import os
import asyncio
from typing import Optional, Callable
import os

class SpeechToText:
    """Speech-to-text utility using Azure Foundry Whisper"""
    
    def __init__(self):
        self.api_key = os.getenv("AZURE_WHISPER_KEY")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """
        Transcribe a local audio file using Azure Foundry Whisper.
        
        Args:
            file_path: Path to the local audio file (supports .wav, .mp3, .m4a, .ogg, etc.)
        
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                return None
            
            # Read the audio file
            with open(file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Create BytesIO object for the audio data
            audio_bytes = io.BytesIO(audio_data)
            
            # Transcribe using Azure Whisper
            text = self._transcribe_with_azure_whisper(audio_bytes)
            return text
            
        except Exception as e:
            print(f"Error transcribing file: {e}")
            return None
    
    
    def _transcribe_with_azure_whisper(self, audio_data: io.BytesIO) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data.getvalue())
                temp_file_path = temp_file.name

            try:
                url = "https://dhruv-mgu6p05m-eastus2.cognitiveservices.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01"
                headers = {"Authorization": f"Bearer {self.api_key}"}

                with open(temp_file_path, "rb") as f:
                    files = {"file": ("audio.wav", f, "audio/wav")}
                    data = {"model": "whisper", "language": "en", "response_format": "text"}
                    response = requests.post(url, headers=headers, files=files, data=data)

                if response.status_code == 200:
                    return response.text.strip()
                else:
                    print(f"Azure Foundry Whisper API Error: {response.status_code} - {response.text}")
                    return None

            finally:
                os.unlink(temp_file_path)

        except Exception as e:
            print(f"Azure Foundry Whisper transcription error: {e}")
            return None
