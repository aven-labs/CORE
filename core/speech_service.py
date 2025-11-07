"""
Speech processing services for the Aven Speech API
"""
import tempfile
import os
import io
from flask import request, jsonify, send_file
from utils.speech_to_text import SpeechToText
from utils.text_to_speech import TextToSpeech


class SpeechService:
    """Service class for speech-to-text and text-to-speech operations"""
    
    def __init__(self):
        self.speech_to_text = SpeechToText()
        self.text_to_speech = TextToSpeech()
    
    def process_speech_to_text(self):
        """Convert audio file to text using Azure Whisper"""
        try:
            if 'audio' not in request.files:
                return jsonify({"error": "No audio file provided"}), 400
            
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"error": "No audio file selected"}), 400
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_file.save(temp_file.name)
                temp_file_path = temp_file.name
            
            try:
                # Read audio data
                with open(temp_file_path, 'rb') as f:
                    audio_data = io.BytesIO(f.read())
                
                # Transcribe using Azure Whisper
                text = self.speech_to_text._transcribe_with_azure_whisper(audio_data)
                
                if text:
                    return jsonify({
                        "success": True,
                        "text": text,
                        "language": "en"
                    })
                else:
                    return jsonify({"error": "No speech detected"}), 400
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            return jsonify({"error": f"Speech recognition failed: {str(e)}"}), 500
    
    def process_text_to_speech(self):
        """Convert text to speech using ElevenLabs API"""
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
            
            text = data['text']
            if not text.strip():
                return jsonify({"error": "Empty text provided"}), 400
            
            # Get voice parameter if provided
            voice = data.get('voice', 'default')
            
            # Generate speech using ElevenLabs API
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                # Use TextToSpeech utility
                audio_file_path = self.text_to_speech._generate_audio_file(text, temp_audio_path, voice)
                
                if audio_file_path and os.path.exists(temp_audio_path):
                    # Return the audio file
                    return send_file(
                        temp_audio_path,
                        mimetype='audio/mpeg',
                        as_attachment=True,
                        download_name='speech.mp3'
                    )
                else:
                    return jsonify({"error": "TTS generation failed"}), 500
                    
            except Exception as e:
                return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500
            finally:
                # Clean up temporary file after sending
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            return jsonify({"error": f"Text-to-speech failed: {str(e)}"}), 500
