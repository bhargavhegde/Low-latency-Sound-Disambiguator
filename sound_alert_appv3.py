# 🎧 Low-latency Sound Disambiguator — Real-Time Audio Intelligence Dashboard
# ⚡ Optimized for minimal detection latency, smooth UI, and stereo awareness.

import streamlit as st
import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv, time, pandas as pd, threading, queue
from datetime import datetime
import plotly.graph_objects as go
import requests

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Low-latency Sound Disambiguator", page_icon="🎧", layout="wide")

st.markdown("""
    <h2 style='font-size:28px; font-weight:700; margin-bottom:0.3rem;'>
        🎧 Low-latency Sound Disambiguator
    </h2>
    <p style='font-size:16px; color:gray; margin-top:0;'>
        Real-time sound detection, AI interpretation, and intelligent visual alerts using <b>YAMNet + Ollama (Mistral)</b>.
    </p>
""", unsafe_allow_html=True)

# ✅ Fix for duplicate faded tab bars
st.markdown("<style>div[data-baseweb='tab-list'] + div[data-baseweb='tab-list']{display:none !important;}</style>", unsafe_allow_html=True)

# -------------------- STATE INIT --------------------
for k in ["hist", "conf", "dirs", "ai", "amplitude", "alert_placeholder"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k != "alert_placeholder" else st.empty()

# -------------------- STATUS --------------------
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
  <div style="width:14px;height:14px;border-radius:50%; background:#22c55e;animation:pulse 1.2s infinite;"></div>
  <span style="font-size:14px;color:#64748b;">Active Listening System — Online</span>
</div>
<style>@keyframes pulse {0%{opacity:0.3;}50%{opacity:1;}100%{opacity:0.3;}}</style>
""", unsafe_allow_html=True)

# -------------------- MODEL LOAD --------------------
@st.cache_resource
def load_yamnet():
    yam = hub.load("https://tfhub.dev/google/yamnet/1")
    csv_path = yam.class_map_path().numpy().decode("utf-8")
    labels = [r["display_name"] for r in csv.DictReader(open(csv_path))]
    return yam, labels

yamnet, labels = load_yamnet()

# -------------------- UTILS --------------------
def classify(audio):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    scores, _, _ = yamnet(audio)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top_idx = tf.argmax(mean_scores).numpy()
    return labels[top_idx], float(mean_scores[top_idx].numpy())

def ai_insight(text):
    try:
        prompt = {"model": "mistral", "prompt": f"Summarize in ≤ 2 lines:\n{text}", "stream": False}
        r = requests.post("http://localhost:11434/api/generate", json=prompt, timeout=15)
        return r.json().get("response", "").strip() if r.status_code == 200 else f"⚠️ Ollama error {r.status_code}"
    except Exception as e:
        return f"⚠️ AI unavailable: {e}"

# -------------------- DIRECTION LOGIC --------------------
def estimate_direction(audio, use_fake=True):
    """Estimate direction using stereo energy balance or simulated angle."""
    if use_fake or audio.ndim < 2:
        return np.random.uniform(-90, 90)
    else:
        left_energy = np.sum(np.abs(audio[:, 0]))
        right_energy = np.sum(np.abs(audio[:, 1]))
        balance = (right_energy - left_energy) / (right_energy + left_energy + 1e-9)
        return np.clip(balance * 90, -90, 90)

# -------------------- AUDIO STREAM --------------------
audio_queue = queue.Queue()
stop_flag = False
MAX_HISTORY = 5

def detect_channels():
    """Auto-detect available input channels (fallback to mono)."""
    try:
        info = sd.query_devices(kind="input")
        channels = info["max_input_channels"]
        return 2 if channels >= 2 else 1
    except Exception:
        return 1

def audio_stream_listener(sr=16000, chunk_dur=0.8, overlap=0.3):
    """Continuously capture audio chunks and enqueue latest ones."""
    global stop_flag
    channels = detect_channels()
    use_fake = channels == 1
    frame_len = int(chunk_dur * sr)
    hop_len = int((chunk_dur - overlap) * sr)
    buffer = np.zeros((frame_len, channels), dtype=np.float32)
    stream = sd.InputStream(samplerate=sr, channels=channels, dtype="float32")
    with stream:
        while not stop_flag:
            data, _ = stream.read(hop_len)
            buffer[:-hop_len] = buffer[hop_len:]
            buffer[-hop_len:] = data
            if not audio_queue.empty():
                audio_queue.get_nowait()  # keep only newest
            audio_queue.put((buffer.copy(), use_fake))

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Settings")
dur = st.sidebar.slider("Detection interval (s)", 0.5, 3.0, 1.0, 0.1)
conf_threshold = st.sidebar.slider("Alert sensitivity (conf %)", 50, 90, 70)
run = st.sidebar.button("▶️ Start Listening")

# -------------------- TABS (Dynamic Only, no duplicates or state errors) --------------------
with st.container():
    tabs = st.tabs(["🎧 Live", "📜 History", "📈 Analytics", "🧠 Insights"])
    live_placeholder = tabs[0].empty()
    history_placeholder = tabs[1].empty()
    analytics_placeholder = tabs[2].empty()
    insights_placeholder = tabs[3].empty()

# -------------------- MAIN LOOP --------------------
if run:
    st.toast("🎧 Listening … Press Ctrl + C to stop.")
    stop_flag = False
    threading.Thread(target=audio_stream_listener, daemon=True).start()

    while True:
        try:
            if audio_queue.empty():
                time.sleep(0.1)
                continue

            # Always process latest buffer only
            while not audio_queue.empty():
                audio, use_fake_dir = audio_queue.get()

            # --- DETECTION & ALERT LOGIC ---
            lab, conf = classify(audio)
            ang = estimate_direction(audio, use_fake=use_fake_dir)
            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S")
            response_id = f"R-{now.strftime('%Y%m%d-%H%M%S')}"
            amp = np.max(np.abs(audio))
            ai = ai_insight(f"Detected {lab} ({conf*100:.1f}%). Provide contextual meaning.")

            CRITICAL_SOUNDS = [
                "fire alarm","siren","explosion","gunshot","crying","baby","dog bark",
                "shout","screaming","glass","car horn","emergency vehicle","alarm","buzzer"
            ]

            if any(c in lab.lower() for c in CRITICAL_SOUNDS):
                badge = "🔴 Alert"
            elif "speech" in lab.lower() or "silence" in lab.lower():
                badge = "🟢 Safe"
            elif conf > conf_threshold / 100:
                badge = "🔴 Alert"
            else:
                badge = "🟡 Neutral"

            # --- UPDATE STATE ---
            event = {"id": response_id, "time": timestamp, "label": lab,
                     "conf": conf, "dir": ang, "amp": amp, "ai": ai, "badge": badge}
            st.session_state["hist"].insert(0, event)
            st.session_state["hist"] = st.session_state["hist"][:5]
            st.session_state["conf"].append(conf)
            st.session_state["dirs"].append(ang)
            st.session_state["amplitude"].append(amp)
            st.session_state["ai"].insert(0, f"{badge} | {ai}")
            st.session_state["ai"] = st.session_state["ai"][:5]

            # --- VISUAL ALERT BANNER ---
            if "🔴" in badge:
                color, border = "#fee2e2", "#ef4444"
                heading = f"🚨 ALERT: {lab.upper()} DETECTED!"
                note = "Immediate attention required."
            elif "🟡" in badge:
                color, border = "#fef9c3", "#eab308"
                heading = f"⚠️ WARNING: {lab.upper()}"
                note = "Monitor surroundings carefully."
            else:
                color, border = "#dcfce7", "#16a34a"
                heading = f"✅ SAFE: {lab.upper()}"
                note = "No immediate action required."

            alert_html = f"""
                <div style="position:fixed;top:100px;right:50px;
                    background:{color};
                    border:3px solid {border};
                    padding:14px 24px;
                    border-radius:12px;
                    width:340px;
                    text-align:left;
                    z-index:9999;
                    box-shadow:0 2px 10px rgba(0,0,0,0.1);
                    animation:flash 1.2s ease-in-out infinite;">
                    <h3 style="margin:0;color:#1e293b;">{heading}</h3>
                    <p style="margin-top:4px;color:#334155;font-size:15px;">{note}</p>
                </div>
                <style>
                @keyframes flash {{
                    0%{{opacity:0.6;}} 50%{{opacity:1;}} 100%{{opacity:0.6;}}
                }}
                </style>
            """
            st.session_state["alert_placeholder"].markdown(alert_html, unsafe_allow_html=True)

            # --- LIVE TAB ---
            with live_placeholder.container():
                st.markdown(f"<h3>🆔 {response_id} • 🕒 {timestamp}</h3>", unsafe_allow_html=True)
                st.markdown(f"### 🎵 {lab} ({conf*100:.1f} %)")
                mode_text = "🧠 Simulated" if use_fake_dir else "🎧 Stereo"
                dir_label = "↖️ Left" if ang < -30 else "↗️ Right" if ang > 30 else "⬆️ Front"
                st.markdown(f"**Direction ({mode_text}):** {dir_label} ({ang:.1f}°)")
                st.markdown(f"**AI Insight:** {ai}")
                st.caption(f"{badge} • Last updated: {timestamp}")

            # --- HISTORY TAB ---
            with history_placeholder.container():
                st.markdown("### 📜 Detection History (Latest 5)")
                df = pd.DataFrame(st.session_state["hist"])
                st.dataframe(df[["id", "time", "label", "conf", "dir", "ai"]],
                             use_container_width=True, hide_index=True, height=250)

            # --- ANALYTICS TAB ---
            with analytics_placeholder.container():
                st.markdown("### 📈 Recent Analytics")
                if len(st.session_state["conf"]) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=st.session_state["conf"], mode="lines+markers", name="Confidence"))
                    fig.add_trace(go.Scatter(y=st.session_state["amplitude"], mode="lines+markers", name="Amplitude"))
                    fig.add_trace(go.Scatter(y=st.session_state["dirs"], mode="lines+markers", name="Direction (°)"))
                    fig.update_layout(height=250, margin=dict(t=10,b=0),
                                      plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc")
                    st.plotly_chart(fig, use_container_width=True)

            # --- INSIGHTS TAB ---
            with insights_placeholder.container():
                st.markdown("### 🧠 AI Summaries (Recent)")
                for s in st.session_state["ai"]:
                    color = "#bbf7d0" if "🟢" in s else "#fecaca" if "🔴" in s else "#fef9c3"
                    border = "#22c55e" if "🟢" in s else "#ef4444" if "🔴" in s else "#eab308"
                    st.markdown(f"""
                        <div style="background:{color};border-left:8px solid {border};
                            padding:12px;border-radius:10px;margin-bottom:8px;
                            box-shadow:0 2px 6px rgba(0,0,0,0.05);">
                            <b>🧩 Summary:</b><br>{s}</div>""", unsafe_allow_html=True)

            time.sleep(0.4)

        except KeyboardInterrupt:
            stop_flag = True
            st.warning("🛑 Stopped listening.")
            break
