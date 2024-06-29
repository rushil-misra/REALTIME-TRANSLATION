
import streamlit as st
import sounddevice as sd
import soundfile as sf
import whisper
from transformers import MarianMTModel, MarianTokenizer

try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    st.error(f"Error loading Whisper model: {e}")

def load_marian_model(src_lang, tgt_lang):
    try:
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading MarianMT model: {e}")
        return None, None

marian_model, marian_tokenizer = load_marian_model("en", "es")

def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, fp16=False)
        return result['text']
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

def translate_text(text, src_lang="en", tgt_lang="es"):
    try:
        inputs = marian_tokenizer(text, return_tensors="pt", padding=True)
        translated = marian_model.generate(**inputs)
        translated_text = marian_tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return ""

def record_audio(duration=5, filename='temp.wav'):
    fs = 44100  
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()  
        sf.write(filename, recording, fs)
        return filename
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

st.title("Real-Time Speech Translation")

st.write("Press the button to start recording your speech for translation.")

if st.button("Start Recording"):
    st.write("Recording...")
    audio_file = record_audio(duration=5) 
    if audio_file:
        st.write("Recording complete")

        st.write("Transcribing audio...")
        transcribed_text = transcribe_audio(audio_file)
        if transcribed_text:
            st.write(f"Transcribed Text: {transcribed_text}")

            st.write("Translating text...")
            translated_text = translate_text(transcribed_text)
            if translated_text:
                st.write(f"Translated Text: {translated_text}")
            else:
                st.error("Translation failed.")
        else:
            st.error("Transcription failed.")
    else:
        st.error("Recording failed.")
