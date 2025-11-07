import os
import dotenv

dotenv.load_dotenv()

# Azure Foundry Whisper Configuration
# Get your API key from Azure Portal
AZURE_WHISPER_KEY = os.getenv("AZURE_WHISPER_KEY")

# ElevenLabs Configuration
# Get your API key from https://elevenlabs.io/
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ElevenLabs Voice Options
# You can get voice IDs from the ElevenLabs dashboard
ELEVENLABS_VOICES = {
    "default": "pNInz6obpgDQGcFmaJgB",  # Adam (default)
    "female": "EXAVITQu4vr4xnSDxMaL",   # Bella
    "male": "pNInz6obpgDQGcFmaJgB",     # Adam
    "young_female": "EXAVITQu4vr4xnSDxMaL",  # Bella
    "young_male": "VR6AewLTigWG4xSOukaG"     # Josh
}

# ElevenLabs Model Configuration
ELEVENLABS_MODEL = "eleven_multilingual_v2"  # or "eleven_monolingual_v1"
ELEVENLABS_STABILITY = 0.5
ELEVENLABS_SIMILARITY_BOOST = 0.5
