"""
Text-to-speech utilities using ElevenLabs API
"""
import pygame
import tempfile
import os
import threading
import requests
import io
from typing import Optional, Callable
from config.elevenlabs_config import (
    ELEVENLABS_API_KEY, 
    ELEVENLABS_VOICES, 
    ELEVENLABS_MODEL,
    ELEVENLABS_STABILITY,
    ELEVENLABS_SIMILARITY_BOOST
)


class TextToSpeech:
    """Text-to-speech utility using ElevenLabs API"""
    
    def __init__(self, api_key: str = None, voice_id: str = None):
        self._is_speaking = False
        self.api_key = api_key or ELEVENLABS_API_KEY
        self.voice_id = voice_id or ELEVENLABS_VOICES["default"]
        
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable.")
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        print("ElevenLabs TTS initialized successfully!")
    
    def _generate_audio_file(self, text: str, output_path: str, voice: str = "default") -> Optional[str]:
        """Generate audio file using ElevenLabs API"""
        try:
            audio_data = self._generate_audio_with_elevenlabs(text, voice)
            
            if audio_data:
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                
                return output_path if os.path.exists(output_path) else None
            else:
                return None
            
        except Exception as e:
            print(f"Audio generation error: {e}")
            return None
    
    def _generate_audio_with_elevenlabs(self, text: str, voice: str = "default") -> Optional[bytes]:
        """Generate audio using ElevenLabs API"""
        try:
            voice_id = ELEVENLABS_VOICES.get(voice, ELEVENLABS_VOICES["default"])
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": {
                    "stability": ELEVENLABS_STABILITY,
                    "similarity_boost": ELEVENLABS_SIMILARITY_BOOST
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"ElevenLabs API request error: {e}")
            return None
