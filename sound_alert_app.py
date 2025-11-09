import streamlit as st
import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import scipy.signal
import csv, time, matplotlib.pyplot as plt
from datetime import datetime
from tdoa_utils import estimate_angle

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Sound Alert & Direction Detector", page_icon="🔊", layout="wide")
st.title("🎧 Real-Time Sound Alert & Direction System")
st.markdown("Detects **sirens**, **dog barks**, and other **danger sounds**, showing simulated direction even on mono mics.")

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
event_log = []  # store detections for UI log

# ---------------- AUDIO FUNCTIONS ----------------
def get_audio_chunk(duration=3, sample_rate=16000):
    """Capture one audio chunk; warn only once."""
    global stereo_checked, stereo_supported

    if not stereo_checked:
        stereo_checked = True
        try:
            sd.rec(int(0.5 * sample_rate), samplerate=sample_rate, channels=2, dtype="float32")
            sd.wait()
            st.sidebar.success("🎙️ Stereo mic detected (direction detection active).")
            print("✅ Stereo mic detected (direction detection active).")
        except sd.PortAudioError:
            stereo_supported = False
            st.sidebar.warning("⚠️ Stereo mic not available — using mono mode with simulated stereo.")
            print("⚠️ Stereo mic not available — using mono mode with simulated stereo.")

    if stereo_supported:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype="float32")
    else:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def simulate_stereo(audio, delay_ms=3, sample_rate=16000):
    """Simulate stereo by adding a tiny delay to right channel."""
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
    "Crying": "https://cdn-icons-png.flaticon.com/512/3654/3654602.png",
    "Scream": "https://cdn-icons-png.flaticon.com/512/4713/4713456.png",
    "Gunshot": "https://cdn-icons-png.flaticon.com/512/5803/5803823.png",
    "Thunder": "https://cdn-icons-png.flaticon.com/512/1146/1146869.png"
}

# ---------------- SETTINGS ----------------
st.sidebar.header("⚙️ Settings")
duration = st.sidebar.slider("Recording duration (seconds)", 1, 5, 3)
mic_spacing = st.sidebar.number_input("Mic spacing (m)", min_value=0.05, max_value=0.5, value=0.15, step=0.01)
simulate = st.sidebar.checkbox("🎭 Enable Fake Stereo Mode", value=True)
run_btn = st.sidebar.button("🎙️ Start Listening")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 2])
log_placeholder = col2.empty()  # detection history
current_output = col1.empty()   # live status

# ---------------- COMPASS PLOT ----------------
def plot_direction(angle):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta = np.deg2rad(angle)
    ax.arrow(theta, 0, 0, 1, width=0.05, color="red")
    ax.set_title(f"Direction: {angle:.1f}°", pad=20)
    st.pyplot(fig)

# ---------------- MAIN LOOP ----------------
if run_btn:
    st.write("🎧 Listening... Press **Stop** in terminal (Ctrl+C) to end.")
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

            # Log and print
            log_entry = f"[{timestamp}] 🎵 {label} ({conf*100:.2f}%)"
            if angle is not None:
                log_entry += f" | 🧭 Direction: {angle:.1f}°"
            print(log_entry)
            event_log.insert(0, log_entry)

            # UI output
            matched = next((k for k in alert_sounds if k.lower() in label.lower()), None)
            with current_output.container():
                st.markdown(f"### 🎵 **Detected:** {label} ({conf*100:.2f}%)")
                if matched:
                    st.error(f"🚨 ALERT: {matched.upper()} detected!")
                    st.image(alert_sounds[matched], width=250)
                    if angle is not None:
                        st.markdown(f"🧭 **Simulated Direction:** {angle:.1f}°")
                        plot_direction(angle)
                else:
                    st.success("✅ Environment safe.")

            # Show full history
            with log_placeholder.container():
                st.markdown("### 🧾 Detection History (most recent first)")
                for log in event_log[:10]:  # show last 10 detections
                    st.text(log)

            time.sleep(1)

        except KeyboardInterrupt:
            st.warning("🛑 Detection stopped by user.")
            print("🛑 Detection stopped by user.")
            break
