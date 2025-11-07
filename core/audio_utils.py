"""
Shared audio utilities and constants
"""
import tempfile
import os


class AudioUtils:
    """Utility class for common audio operations"""
    
    @staticmethod
    def create_temp_audio_file(suffix: str = ".wav") -> tuple[str, str]:
        """Create a temporary audio file and return (file_path, temp_file_name)"""
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        return temp_file_path, os.path.basename(temp_file_path)
    
    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """Clean up a temporary file if it exists"""
        if os.path.exists(file_path):
            os.unlink(file_path)
    
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """Validate that an audio file exists and is readable"""
        return os.path.exists(file_path) and os.path.isfile(file_path)


# Audio constants
SAMPLE_RATE_CHATTTS = 24000  # ChatTTS uses 24kHz sample rate
AUDIO_FORMAT_WAV = "audio/wav"
TEMP_FILE_SUFFIX_WAV = ".wav"
