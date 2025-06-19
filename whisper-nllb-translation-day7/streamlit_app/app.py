import os
import sys
import re
import json
import io
from datetime import datetime

import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    from docx import Document
except ImportError:
    Document = None

torch.classes.__path__ = []

# ----- Translator class -----
class Translator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/nllb-200-1.3B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        self.lang_code_map = {
            ("en", "zh"): ("eng_Latn", "zho_Hans"),
            ("zh", "en"): ("zho_Hans", "eng_Latn"),
        }

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if (src_lang, tgt_lang) not in self.lang_code_map:
            raise ValueError(f"Unsupported language pair: {src_lang}-{tgt_lang}")

        src_code, tgt_code = self.lang_code_map[(src_lang, tgt_lang)]
        self.tokenizer.src_lang = src_code

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return output

translator = Translator()

# ----- Helper functions -----
def detect_language(text: str) -> str:
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total = len(text.replace(' ', ''))
    if total == 0:
        return "unknown"
    return "zh" if (chinese_chars / total) > 0.3 else "en"

def clean_text(text: str) -> str:
    return re.sub(r"\W{3,}", " ", text).strip()

def save_json(data: dict, folder: str, prefix: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (" ", "_", "-")).rstrip()
    filepath = os.path.join(folder, f"{ts}_{prefix}_{safe_filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath

def save_csv(data: list, folder: str, prefix: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (" ", "_", "-")).rstrip()
    filepath = os.path.join(folder, f"{ts}_{prefix}_{safe_filename}.csv")
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding="utf-8")
    return filepath

def save_audio_results(result: dict, translation: str, src: str, tgt: str, filename: str):
    out = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "filename": filename,
        "source_language": src,
        "target_language": tgt,
        "full_transcript": result.get("text", ""),
        "full_translation": translation,
        "segments": [],
        "word_details": []
    }
    for seg in result.get("segments", []):
        seg_data = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
            "confidence": seg.get("confidence", 0),
            "words": []
        }
        for w in seg.get("words", []):
            wd = {
                "word": w.get("text"),
                "start": w.get("start"),
                "end": w.get("end"),
                "confidence": w.get("confidence", 0)
            }
            seg_data["words"].append(wd)
            out["word_details"].append(wd)
        out["segments"].append(seg_data)

    json_path = save_json(out, "outputs/audio_translations", "translation", filename)
    csv_path = None
    if out["word_details"]:
        csv_path = save_csv(out["word_details"], "outputs/audio_translations", "words", filename)
    return json_path, csv_path

def save_text_results(original: str, translation: str, src: str, tgt: str):
    out = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_language": src,
        "target_language": tgt,
        "original_text": original,
        "translation": translation,
        "detected_language": detect_language(original)
    }
    json_path = save_json(out, "outputs/text_translations", "text_translation", "text")
    return json_path

def translate_text_only(text: str, src: str, tgt: str) -> str:
    txt = clean_text(text)
    try:
        from utils.context_optimizer import optimize_context
        txt = optimize_context(txt, src, tgt)
    except ImportError:
        pass
    try:
        return translator.translate(txt, src, tgt)
    except Exception as e:
        return f"[Translation error: {e}]"

def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            if PyPDF2 is None:
                return "[PyPDF2 not installed]"
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            return "".join(page.extract_text() or "" for page in reader.pages)
        elif uploaded_file.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
            if Document is None:
                return "[python-docx not installed]"
            doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return "[Unsupported file type]"
    except Exception as e:
        return f"[File read error: {e}]"

# ----- Streamlit UI -----
st.set_page_config(page_title="English ↔ Chinese Speech & Text Translator", layout="wide")

st.markdown("""
<style>
* {
    font-family: "Microsoft YaHei", "SimSun", "Noto Sans CJK SC", sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.title("English ↔ Chinese Speech & Text Translator")

tab1, tab2, tab3 = st.tabs(["🎵 Audio Translation", "📝 Text/File Translation", "📁 Saved Results"])

with tab1:
    st.header("Audio → Text & Translation")

    src = st.selectbox("Source Language", ["en", "zh"], key="audio_src")
    tgt = "zh" if src == "en" else "en"

    audio_file = st.file_uploader("Upload audio (wav, mp3, m4a)", type=["wav", "mp3", "m4a"], key="audio_uploader")

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Transcribe & Translate Audio"):
            with st.spinner("Transcribing & translating..."):
                try:
                    from utils.whisper_integration import transcribe_audio
                    result = transcribe_audio(audio_file)
                except ImportError:
                    result = {"text": "[Transcription not implemented]"}
                transcript_text = result.get("text", "") if isinstance(result, dict) else result
                detected = detect_language(transcript_text)
                translation = translate_text_only(transcript_text, src, tgt)
                json_file, csv_file = save_audio_results(result if isinstance(result, dict) else {"text": transcript_text}, translation, src, tgt, audio_file.name)

            st.success("Done!")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Detected Language:** {detected.upper()}")
                st.text_area("Transcript:", transcript_text, height=150, disabled=True)
            with c2:
                st.markdown(f"**Translation ({tgt.upper()}):**")
                st.markdown(
                    f"<div style='font-family:Microsoft YaHei; white-space:pre-wrap;'>{translation}</div>",
                    unsafe_allow_html=True
                )

            if csv_file:
                st.markdown("### Word-Level Confidence")
                df = pd.read_csv(csv_file)
                st.dataframe(df, use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                with open(json_file, encoding="utf-8") as f:
                    json_data = f.read()
                st.download_button("Download JSON", data=json_data, file_name=os.path.basename(json_file), mime="application/json")
            with dl2:
                if csv_file:
                    with open(csv_file, encoding="utf-8") as f:
                        csv_data = f.read()
                    st.download_button("Download CSV", data=csv_data, file_name=os.path.basename(csv_file), mime="text/csv")

with tab2:
    st.header("Direct Text & File Translation")

    col1, col2 = st.columns(2)
    with col1:
        text_src = st.selectbox("Source Language", ["en", "zh"], key="text_src")
    with col2:
        text_tgt = st.selectbox("Target Language", ["zh", "en"], key="text_tgt")

    txt_input = st.text_area("Enter text:", height=150)
    uploaded_file = st.file_uploader("Upload text file (txt, pdf, docx)", type=["txt", "pdf", "docx"], key="text_file_uploader")

    file_txt = read_uploaded_file(uploaded_file)
    combined_text = (txt_input + "\n" + file_txt).strip() if file_txt else txt_input.strip()

    if st.button("Translate Text"):
        if combined_text:
            with st.spinner("Translating..."):
                detected = detect_language(combined_text)
                translation = translate_text_only(combined_text, text_src, text_tgt)
                json_file = save_text_results(combined_text, translation, text_src, text_tgt)
            st.success("Translated!")
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f"**Detected:** {detected.upper()}")
                st.text_area("Original:", combined_text, height=200, disabled=True)
            with t2:
                st.markdown(f"**Translation ({text_tgt.upper()}):**")
                st.markdown(
                    f"<div style='font-family:Microsoft YaHei; white-space:pre-wrap;'>{translation}</div>",
                    unsafe_allow_html=True
                )
                st.download_button("Download Translation", data=translation,
                                   file_name=f"translation_{text_src}_to_{text_tgt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                   mime="text/plain")
        else:
            st.warning("Please enter or upload text to translate.")

with tab3:
    st.header("Saved Results")

    st.subheader("Audio Translations")
    audio_dir = "outputs/audio_translations"
    os.makedirs(audio_dir, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".json")]
    if audio_files:
        selected_audio = st.selectbox("Select audio result:", audio_files)
        with open(os.path.join(audio_dir, selected_audio), encoding="utf-8") as f:
            data = json.load(f)
        a1, a2 = st.columns(2)
        with a1:
            st.text_area("Transcript:", data.get("full_transcript", ""), height=150, disabled=True)
        with a2:
            st.markdown("**Translation:**")
            st.markdown(
                f"<div style='font-family:Microsoft YaHei; white-space:pre-wrap;'>{data.get('full_translation', '')}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No saved audio translations found.")

    st.subheader("Text Translations")
    text_dir = "outputs/text_translations"
    os.makedirs(text_dir, exist_ok=True)
    text_files = [f for f in os.listdir(text_dir) if f.endswith(".json")]
    if text_files:
        selected_text = st.selectbox("Select text result:", text_files)
        with open(os.path.join(text_dir, selected_text), encoding="utf-8") as f:
            data = json.load(f)
        b1, b2 = st.columns(2)
        with b1:
            st.text_area("Original:", data.get("original_text", ""), height=200, disabled=True)
        with b2:
            st.markdown("**Translation:**")
            st.markdown(
                f"<div style='font-family:Microsoft YaHei; white-space:pre-wrap;'>{data.get('translation', '')}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No saved text translations found.")

with st.sidebar:
    st.markdown("### Features")
    st.markdown("""
- Audio transcription & translation with word-level confidence and timestamps  
- Direct text input & file upload (TXT, PDF, DOCX)  
- Idiom-aware preprocessing (optional)  
- Hugging Face NLLB MT model translation  
- JSON/CSV exports and downloads  
""")
    st.markdown("### Requirements")
    st.code("pip install streamlit torch transformers pandas PyPDF2 python-docx", language="bash")
