# Low-latency-Sound-Disambiguator
# 🔊 Low-Latency Sound Disambiguator  
### Real-Time Sound Classification, Direction Estimation, and AI Summarization

This project is a real-time **Sound Alert & Direction Detection System** built for accessibility and situational awareness — such as aiding hearing-impaired individuals or detecting danger sounds like **sirens**, **dog barks**, **explosions**, or **gunshots** in real environments.

It integrates **deep learning**, **signal processing**, and **generative AI** to perform:
- 🎧 **Real-time sound classification** using [Google YAMNet](https://tfhub.dev/google/yamnet/1)
- 🧭 **Sound direction estimation** using TDOA (Time Difference of Arrival)
- 🎭 **Simulated stereo mode** for mono microphones (MacBook support)
- 🧠 **AI-generated natural summaries** using the open-source **Mistral model via Ollama**
- 🌗 **Streamlit UI** with dark/light themes, visual alerts, and trend analysis

---

## 🧩 Features Overview

| Feature | Description |
|----------|-------------|
| 🎙️ Live Sound Classification | Real-time YAMNet model inference for 521 sound classes |
| 🧭 Direction Estimation | Calculates 2D sound angle using cross-correlation (TDOA) |
| 🎭 Simulated Stereo | Adds micro-delay to mono streams for direction simulation |
| 🧠 AI Summary | Uses local Mistral LLM to describe detected sounds intelligently |
| 📜 Detection History | Scrollable log of latest sound detections |
| 📈 Confidence Trend | Rolling chart of YAMNet model confidence |
| 🌗 Dual Themes | Toggle between dark and light UI modes |
| 🧾 Database (Planned) | Automatic event logging with SQLite for analytics |

---

## ⚙️ Installation & Execution Guide

### 🧱 Prerequisites
- macOS (optimized for M1/M2 devices)
- Python ≥ 3.9
- [Homebrew](https://brew.sh/) for installing Ollama
- Internet connection (for first-time model downloads)

---

### 🪄 Step 1 — Clone Repository
```bash
git clone https://github.com/<your-username>/Low-Latency-Sound-Disambiguator.git
cd Low-Latency-Sound-Disambiguator
```

---

### 🧰 Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

#### 📦 requirements.txt
```
streamlit
sounddevice
numpy
tensorflow
tensorflow-hub
scipy
matplotlib
pandas
```

---

### 🤖 Step 3 — Install Ollama (for AI Summaries)
```bash
brew install ollama
```
Then launch Ollama (background service):
```bash
open -a Ollama
```
Pull the **Mistral** model:
```bash
ollama pull mistral
```
✅ Test Ollama locally:
```bash
ollama run mistral "Summarize this: A siren and dog bark were heard."
```

---

### 🚀 Step 4 — Run the App
```bash
streamlit run sound_alert_app.py
```
Then open in your browser:
```
http://localhost:8501
```

---

## 🧭 Interface Overview

| Section | Description |
|----------|-------------|
| **🎧 Live Detection** | Real-time detection with confidence bar and direction compass |
| **📜 History** | Timestamped detection logs |
| **📈 Confidence Trend** | Rolling chart of last N predictions |
| **🧠 AI Summary** | Short contextual summary from last 10 detections |
| **🌗 Sidebar** | Theme toggle, duration, mic spacing, stereo simulation |

---

## 🧠 Technology Stack

| Layer | Library / Tool | Purpose |
|--------|----------------|----------|
| Frontend | Streamlit | Interactive web UI |
| Classification | TensorFlow Hub (YAMNet) | Sound recognition |
| DSP | SciPy / NumPy | Resampling, cross-correlation |
| Direction Estimation | Custom (TDOA) | Angle computation |
| AI Summary | Ollama + Mistral | Local LLM summaries |
| Visualization | Matplotlib | Polar direction plots |
| Data Storage (Planned) | SQLite | Persistent event logs |

---

## 🧰 Project Structure

```
Low-Latency-Sound-Disambiguator/
│
├── sound_alert_app.py        # Streamlit app entry point
├── tdoa_utils.py             # Audio direction (TDOA) computation
├── requirements.txt          # Dependencies list
├── setup.sh                  # Automated setup script
├── README.md                 # Full documentation
└── IMPLEMENTATION_DOC.md     # (Merged below in this README)
```

---

## 🧾 Quick Setup (Optional)
```bash
chmod +x setup.sh
./setup.sh
```

#### setup.sh
```bash
#!/bin/bash
echo "🔧 Setting up environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if ! command -v ollama &> /dev/null
then
    echo "⚠️ Installing Ollama..."
    brew install ollama
fi
ollama pull mistral
echo "✅ Done! Run 'streamlit run sound_alert_app.py'"
```

---

## 🧠 Implementation Details

### 1️⃣ Core Concept
The system continuously listens, processes 3-second chunks of audio, classifies sounds, estimates the direction, and generates a live dashboard + AI summary.

### 2️⃣ Module Breakdown
- **sound_alert_app.py** — handles recording, prediction, UI, summaries
- **tdoa_utils.py** — TDOA-based angle estimation
- **AI Summary** — uses Mistral LLM via Ollama for context insights
- **Visualization** — Streamlit dashboard + polar compass

### 3️⃣ Execution Flow
```text
Audio Stream → Preprocess → YAMNet Classification → 
   ↳ Determine Confidence → 
   ↳ Detect Alerts → 
   ↳ Compute Direction → 
   ↳ Display in UI → 
   ↳ Summarize via Mistral
```

### 4️⃣ Example Output
```
🎵 Detected: Siren (94.21%)
🚨 ALERT: SIREN detected!
🧭 Direction: 45.2°
🕒 Last updated: 22:13:10
```

### 5️⃣ Future Enhancements
- 📊 SQLite analytics dashboard  
- 🔔 Voice/TTS feedback  
- ☁️ Cloud integration (Vertex AI / HuggingFace)  
- 📱 Mobile Streamlit version  

---

## 🙌 Credits & Acknowledgements

| Resource | Description |
|-----------|-------------|
| [Google YAMNet](https://tfhub.dev/google/yamnet/1) | Pretrained environmental sound model |
| [TensorFlow Hub](https://tfhub.dev/) | Model repository |
| [Streamlit](https://streamlit.io/) | Dashboard framework |
| [Ollama](https://ollama.com/) | Local LLM runtime |
| [Mistral](https://mistral.ai/) | LLM used for AI summaries |
| [Flaticon](https://www.flaticon.com/) | Alert icons |
| [NumPy / SciPy](https://scipy.org/) | DSP tools |
| **Developed by:** [Rohith Sagar Karnala](https://github.com/rohitsagar363) | University at Buffalo, MS CSE |

---

## 🧾 License
MIT License © 2025 [Rohith Sagar Karnala](https://github.com/rohitsagar363)
