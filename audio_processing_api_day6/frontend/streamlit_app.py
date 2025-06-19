import streamlit as st
import requests
import base64

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Audio Processing POC", layout="wide")
st.title("🎤🔄 Audio Processing POC Demo")

# ---- LOGIN SECTION ----
st.sidebar.header("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
token = st.session_state.get("token", "")

if st.sidebar.button("Login"):
    login_payload = {"username": username, "password": password}
    try:
        login_res = requests.post(f"{API_BASE}/login", data=login_payload)
        if login_res.ok:
            token = login_res.json().get("access_token", "")
            st.session_state["token"] = token
            st.sidebar.success("Logged in!")
        else:
            st.sidebar.error("Login failed. Check credentials.")
    except Exception as e:
        st.sidebar.error(f"Login request failed: {e}")

headers = {"Authorization": f"Bearer {token}"} if token else {}

tab1, tab2, tab3 = st.tabs(["Transcribe", "Translate", "Speak"])

# 1. ASR /transcribe
with tab1:
    st.header("1️⃣ Speech-to-Text (ASR)")
    uploaded = st.file_uploader("Upload audio for transcription", type=["wav", "mp3", "ogg", "flac"])
    can_transcribe = bool(token and uploaded)
    if st.button("Transcribe", disabled=not can_transcribe):
        files = {"audio": (uploaded.name, uploaded.read(), uploaded.type)}
        try:
            res = requests.post(f"{API_BASE}/transcribe", files=files, headers=headers)
            if res.ok:
                data = res.json()
                st.success("✅ Transcription complete")
                st.code(data.get("text", ""))
                if data.get("words"):
                    st.subheader("Word-level Timestamps & Confidence")
                    st.json(data["words"])
                st.info(f"Language: {data.get('language', '')} | Model: {data.get('model_info', {}).get('model_name', '')}")
            else:
                st.error(f"Transcription failed: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    elif uploaded and not token:
        st.warning("Please login to use this feature.")

# 2. MT /translate
with tab2:
    st.header("2️⃣ Text Translation (MT)")
    src_text = st.text_area("Enter text to translate")
    target_lang = st.text_input("Target language code", value="fr")
    can_translate = bool(token and src_text)
    if st.button("Translate", disabled=not can_translate):
        payload = {
            "text": src_text,
            "target_language": target_lang,
            "source_language": "en"
        }
        try:
            res = requests.post(f"{API_BASE}/translate", json=payload, headers=headers)
            if res.ok:
                data = res.json()
                st.success("✅ Translation complete")
                st.write(f"**From ({data.get('source_language', '')}) → To ({data.get('target_language', '')})**")
                st.code(data.get("translated_text", ""))
            else:
                st.error(f"Translation failed: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    elif src_text and not token:
        st.warning("Please login to use this feature.")

# 3. TTS /speak
with tab3:
    st.header("3️⃣ Speech Synthesis (Speak)")
    uploaded_ref = st.file_uploader("Upload reference sample (for voice, optional)", type=["wav", "mp3", "ogg", "flac"], key="ref_audio")
    text_to_speak = st.text_area("Text to synthesize")
    target_lang_speak = st.text_input("Target language code", value="fr", key="speak_lang")
    can_speak = bool(token and text_to_speak)
    if st.button("Speak", disabled=not can_speak):
        files = {}
        if uploaded_ref:
            files["reference_audio"] = (uploaded_ref.name, uploaded_ref.read(), uploaded_ref.type)
        data = {"text": text_to_speak, "target_lang": target_lang_speak}
        try:
            res = requests.post(f"{API_BASE}/speak", data=data, files=files if files else None, headers=headers)
            if res.ok:
                resp = res.json()
                audio_b64 = resp.get("audio_data", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    st.audio(audio_bytes, format="audio/wav")
                st.success("Speech synthesis complete.")
                if resp.get("processing_time"):
                    st.write(f"Processing time: {resp.get('processing_time', 0):.2f}s")
                if resp.get("quality_rating"):
                    st.info(f"Quality: {resp['quality_rating']}")
            else:
                st.error(f"Speech synthesis failed: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    elif text_to_speak and not token:
        st.warning("Please login to use this feature.")

st.sidebar.title("About / Integrated Modules")
st.sidebar.markdown("""
- **ASR**: Whisper (faster-whisper), Vosk (offline/Windows)
- **Translation**: MarianMT (Hugging Face), OpenNMT (optional)
- **TTS & Voice Cloning**: XTTS/YourTTS, ElevenLabs (optional)
- **Speaker Similarity**: Resemblyzer (if supported by backend)
- **Features**: Word-level timestamps, confidence, language detection, model info, error handling
""")
