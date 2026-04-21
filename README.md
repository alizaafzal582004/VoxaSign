<h1 align="center"> Voxasign </h1>
<h3 align="center">AI-Powered Real-Time Sign Language → Text → Speech</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow.js-WebGL-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/MediaPipe-Vision-green?style=for-the-badge&logo=google">
  <img src="https://img.shields.io/badge/Edge--AI-Privacy--First-black?style=for-the-badge">
  <img src="https://img.shields.io/badge/Deployment-Live--Ready-brightgreen?style=for-the-badge">
</p>

<p align="center">
  <b>Bridging Communication Gaps with Inclusive AI 🤝</b><br>
  Real-time gesture translation directly in your browser — no servers, no latency.
</p>

## 📌 Project Overview

**VoxaSign** is a high-performance **Gesture Studio SaaS Dashboard** built to empower the Deaf and Hard of Hearing (DHH) community.

It evolved from a **Python-based deep learning model** into a **fully browser-based Edge-AI system**, delivering:

- ⚡ Real-time predictions  
- 🔒 Complete privacy (on-device processing)  
- 🌐 Cross-platform accessibility  

---

## ⚠️ The Problem

> 🌍 Over **70 million people** rely on sign language globally.

Yet, communication barriers still exist in:

- 🏥 Healthcare  
- 🎓 Education  
- 🏦 Banking  

💡 **VoxaSign solves this** with a **hardware-agnostic, browser-based AI translator**.

---

## ✨ Studio Features

- 🎥 **Live Studio Mode** — Real-time gesture recognition  
- 🖼️ **Photo Analysis Mode** — Upload & analyze static images  
- 🔊 **Speech Engine** — Web Speech API integration  
- 📊 **Session Analytics** — Confidence, counts, stats  
- 🎨 **Premium UI** — Glassmorphism + Warm Paper Theme  
- ⚡ **30+ FPS Performance** with WebGL acceleration  

---

## 🧠 Technical Architecture

### 🔄 Model Conversion (Python → Web)

- Trained using **TensorFlow / Keras**
- Converted to **TensorFlow.js (JSON format)**

**Pipeline:**
```
Camera → Hand Tracking → Landmark Processing → DNN → Text → Speech
```

---

### 🌐 Edge-AI Deployment

- 🔒 **Privacy-First:** No cloud, no data sharing  
- ⚡ **Zero Latency:** Runs entirely in-browser  
- 📱 **Universal:** Desktop + Mobile optimized  

---

## 📂 Project Structure

```
VoxaSignAI/
├── web_model/
│   ├── model.json
│   └── group1-shard1of1.bin
├── index.html
├── style.css
├── script.js
├── realtime_sign.py
├── train_signspeak.py
└── label_map.npy
```

---

## 🚀 Quick Start (Web Studio)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Atiqumer/SignSpeakAI.git
cd VoxaSign
```

### 2️⃣ Run Locally

⚠️ Requires a local server (ES Modules)

**Option 1 — VS Code**
- Install **Live Server**
- Click **Go Live**

**Option 2 — Python Server**
```bash
python -m http.server 8000
```

👉 Open: `http://localhost:8000`

---

## 💻 Python Backend (Training / Legacy)

| Step | Command |
|------|--------|
| Create venv | `python -m venv venv` |
| Activate | `source venv/bin/activate` or `venv\Scripts\activate` |
| Install deps | `pip install -r requirements.txt` |
| Run | `python realtime_sign.py` |

---

## ⚙️ Core Engineering Highlights

- 🎯 **Wrist-Relative Normalization** → Improves accuracy across distances  
- 🔍 **Confidence Threshold (75%)** → Reduces jitter  
- 🧠 **Efficient Memory Handling** → Prevents browser leaks (`tf.dispose`)  
- ⚡ **GPU Acceleration (WebGL)** → Smooth real-time inference  

---

## 🛤️ Roadmap

- [ ] 🧠 NLP-based sentence prediction  
- [ ] 🔁 Two-way communication (Speech → Sign)  
- [ ] 📱 Progressive Web App (Offline Mode)  
- [ ] 🌍 Multi-language support  

---
