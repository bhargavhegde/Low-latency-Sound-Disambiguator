import streamlit as st
import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import scipy.signal
import csv, time, matplotlib.pyplot as plt, base64, subprocess, json
from datetime import datetime
from tdoa_utils import estimate_angle
from collections import Counter

import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Sound Alert & AI Direction Detector", page_icon="🔊", layout="wide")

st.sidebar.markdown("## 🌗 Theme Settings")
dark_mode = st.sidebar.toggle("Enable Dark Mode", value=False)

if dark_mode:
    bg_color = "#0f172a"; text_color = "#f1f5f9"; accent = "#3b82f6"
else:
    bg_color = "#f8fafc"; text_color = "#0f172a"; accent = "#2563eb"

st.markdown(f"""
<style>
    .main {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Segoe UI', sans-serif;
    }}
    .stButton>button {{
        background-color: {accent};
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
    }}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Real-Time Sound Alert & Direction System with 🧠 AI Insights")
st.markdown("Detects **sirens**, **dog barks**, and other **danger sounds**, showing direction and AI-generated summaries.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])
    return model, class_names

with st.spinner("Loading YAMNet model..."):
    model, class_names = load_model()
st.success(f"✅ Model loaded with {len(class_names)} classes.")

# ---------------- GLOBAL FLAGS ----------------
stereo_checked = False
stereo_supported = True
event_log, conf_history, label_history = [], [], []

# ---------------- AUDIO FUNCTIONS ----------------
def get_audio_chunk(duration=3, sample_rate=16000):
    global stereo_checked, stereo_supported
    if not stereo_checked:
        stereo_checked = True
        try:
            sd.rec(int(0.5 * sample_rate), samplerate=sample_rate, channels=2, dtype="float32")
            sd.wait()
            st.sidebar.success("🎙️ Stereo mic detected (direction detection active).")
        except sd.PortAudioError:
            stereo_supported = False
            st.sidebar.warning("⚠️ Stereo mic not available — using mono mode with simulated stereo.")
    if stereo_supported:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype="float32")
    else:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def simulate_stereo(audio, delay_ms=3, sample_rate=16000):
    if audio.ndim > 1 and audio.shape[1] == 2:
        return audio
    delay = int(sample_rate * delay_ms / 1000)
    left = audio
    right = np.concatenate([np.zeros(delay), audio[:-delay]])
    return np.stack([left, right], axis=1)

def preprocess_audio(audio, original_sr=16000, target_sr=16000):
    if original_sr != target_sr:
        n = round(audio.shape[0] * float(target_sr) / original_sr)
        audio = scipy.signal.resample(audio, n, axis=0)
    return audio

def predict_audio(audio):
    mono = np.mean(audio, axis=1)
    scores, _, _ = model(mono)
    mean_scores = tf.reduce_mean(scores, axis=0)
    i = tf.argmax(mean_scores).numpy()
    return class_names[i], mean_scores[i].numpy()

# ---------------- ALERT IMAGES ----------------
alert_sounds = {
    "Siren": "https://cdn-icons-png.flaticon.com/512/808/808561.png",
    "Dog bark": "https://cdn-icons-png.flaticon.com/512/616/616408.png",
    "Fire alarm": "https://cdn-icons-png.flaticon.com/512/4713/4713456.png",
    "Smoke alarm": "https://cdn-icons-png.flaticon.com/512/1643/1643992.png",
    "Explosion": "https://cdn-icons-png.flaticon.com/512/1356/1356436.png",
    "Gunshot": "https://cdn-icons-png.flaticon.com/512/5803/5803823.png"
}

# ---------------- OPEN-SOURCE AI SUMMARY (OLLAMA) ----------------
def generate_ai_summary(labels, confs):
    if not labels:
        return "No sounds detected yet."

    prompt = f"""
You are an intelligent assistant summarizing environmental sound events.
Recent detections: {labels[-10:]}
Confidence values: {['{:.1f}%'.format(c*100) for c in confs[-10:]]}
Summarize what's happening concisely:
- Identify common sounds
- Mention alert types (sirens, dogs, explosions, etc.)
- Assess whether environment seems safe or dangerous
Give a short natural-language summary.
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout.strip()
        return output if output else "⚠️ AI model did not return output."
    except Exception as e:
        return f"⚠️ Could not generate AI summary: {e}"

# ---------------- SETTINGS ----------------
st.sidebar.header("⚙️ Detection Settings")
duration = st.sidebar.slider("Recording duration (seconds)", 1, 5, 3)
mic_spacing = st.sidebar.number_input("Mic spacing (m)", min_value=0.05, max_value=0.5, value=0.15, step=0.01)
simulate = st.sidebar.checkbox("🎭 Enable Fake Stereo Mode", value=True)
run_btn = st.sidebar.button("🎙️ Start Listening")

# ---------------- LAYOUT ----------------
tab1, tab2, tab3, tab4 = st.tabs(["🎧 Live Detection", "📜 History", "📈 Confidence Trend", "🧠 AI Summary"])

# ---------------- MAIN LOOP ----------------
if run_btn:
    st.toast("🎧 Listening... Press Stop in terminal (Ctrl+C) to end.")
    SAMPLE_RATE = 16000

    while True:
        try:
            audio = get_audio_chunk(duration, SAMPLE_RATE)
            if not stereo_supported and simulate:
                audio = simulate_stereo(np.squeeze(audio), delay_ms=np.random.randint(2, 6))
            audio = preprocess_audio(audio, SAMPLE_RATE)

            label, conf = predict_audio(audio)
            angle = estimate_angle(audio, mic_spacing, SAMPLE_RATE)
            timestamp = datetime.now().strftime("%H:%M:%S")

            conf_history.append(conf)
            label_history.append(label)
            if len(conf_history) > 20: conf_history.pop(0)
            if len(label_history) > 50: label_history.pop(0)

            log_entry = f"[{timestamp}] 🎵 {label} ({conf*100:.2f}%)"
            if angle is not None: log_entry += f" | 🧭 Direction: {angle:.1f}°"
            event_log.insert(0, log_entry)
            print(log_entry)

            # --- Live Detection Tab ---
            with tab1:
                st.markdown(f"## 🎵 **Detected:** `{label}` ({conf*100:.2f}%)")
                st.progress(int(conf * 100))
                matched = next((k for k in alert_sounds if k.lower() in label.lower()), None)
                if matched:
                    st.error(f"🚨 ALERT: {matched.upper()} detected!")
                    st.image(alert_sounds[matched], width=180)
                    if angle is not None:
                        st.markdown(f"🧭 **Direction:** {angle:.1f}°")
                else:
                    st.success("✅ Environment safe.")
                st.caption(f"🕒 Last updated: {timestamp}")

            # --- History Tab ---
            with tab2:
                st.markdown("### 📜 Detection Log (latest first)")
                for log in event_log[:15]:
                    st.text(log)

            # --- Confidence Chart ---
            with tab3:
                st.line_chart(conf_history, height=250)

            # --- AI Summary Tab ---
            with tab4:
                with st.spinner("Generating AI summary..."):
                    ai_text = generate_ai_summary(label_history, conf_history)
                st.markdown("### 🧠 AI Summary (powered by Mistral)")
                st.write(ai_text)

            time.sleep(2)

        except KeyboardInterrupt:
            st.warning("🛑 Detection stopped by user.")
            break
