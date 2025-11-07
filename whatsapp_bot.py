import tempfile
import json
from flask import Flask, request, jsonify
import requests
import dotenv
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) 
if current_dir not in sys.path: 
    sys.path.insert(0, current_dir) 
brain_src_path = os.path.join(current_dir, 'brain', 'src') 
brain_src_path = os.path.abspath(brain_src_path) 
if brain_src_path not in sys.path: sys.path.insert(1, brain_src_path) 
from utils.speech_to_text import SpeechToText
from orchestration.orchestrator import AgentOrchestrator
import asyncio
dotenv.load_dotenv()
app = Flask(__name__)
VERIFY_TOKEN = os.getenv("WWHATSAPP_VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("WWHATSAPP_APITOKEN")
PHONE_NUMBER_ID = os.getenv("WWHATSAPP_PHONE_NUMBER_ID")


@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("Webhook verified ✅")
        return challenge
    else:
        return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    try:
        entry = data.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})

        # Handle incoming messages
        messages = value.get("messages")
        if messages:
            message = messages[0]
            from_number = message["from"]
            msg_type = message.get("type")

            if msg_type == "text":
                user_text = message["text"]["body"]

            elif msg_type == "audio":
                media_id = message["audio"]["id"]
                user_text = transcribe_audio(media_id)  # speech-to-text
            else:
                print(f"⚠️ Unsupported message type: {msg_type}")
                user_text = None

            if user_text:
                orchestrator = AgentOrchestrator(user_id=from_number)
                agent_response = orchestrator.start_chat(user_text)
                send_whatsapp_message(from_number, agent_response)

        # Handle message statuses (delivered/read)
        elif value.get("statuses"):
            status_info = value["statuses"][0]
            print("📬 Status update:",
                  status_info["status"], "for", status_info["id"])

        else:
            print("⚠️ Unknown webhook structure:", json.dumps(value, indent=2))

    except Exception as e:
        print("❌ Error processing webhook:", e)

    return jsonify({"status": "ok"}), 200


def transcribe_audio(media_id: str) -> str:
    try:
        media_url_resp = requests.get(
            f"https://graph.facebook.com/v20.0/{media_id}",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
        )
        media_url = media_url_resp.json().get("url")
        audio_resp = requests.get(
            media_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio:
            temp_audio.write(audio_resp.content)
            audio_path = temp_audio.name

        text = SpeechToText().transcribe_file(audio_path)
        return text

    except Exception as e:
        print("❌ Error in audio transcription:", e)
        return None


def send_whatsapp_message(to, message):
    """Send reply to WhatsApp"""
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, json=payload)
    print("📤 WhatsApp API response:", response.status_code, response.text)


if __name__ == '__main__':
    print("Starting WhatsApp server with voice support...")
    app.run(host='0.0.0.0', port=5000, debug=False)
