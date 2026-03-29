# SignSpeak AI: Real-Time Sign Language to Speech Translator

SignSpeak AI is a computer vision and machine learning application designed to bridge the communication gap for the hearing and speech-impaired. The system utilizes MediaPipe for high-fidelity hand landmark detection and a custom-trained TensorFlow model to recognize sign language alphabets and convert them into audible speech in real-time.

## ✨ Features
* **Real-Time Detection**: Processes video frames instantly to identify hand gestures.
* **Coordinate Normalization**: Uses wrist-centric and zoom-invariant scaling to ensure high accuracy regardless of hand position or distance from the camera.
* **Text-to-Speech (TTS)**: Automatically announces the recognized letter using the Google Text-to-Speech (gTTS) library.
* **High Accuracy**: Optimized Neural Network architecture achieving over 98% validation accuracy on the alphabet dataset.

## 🛠️ Tech Stack
* **Language**: Python 3.10.11
* **Hand Tracking**: MediaPipe (Vision Tasks API)
* **Deep Learning**: TensorFlow / Keras
* **Computer Vision**: OpenCV
* **Audio**: gTTS (Google Text-to-Speech)

## 🚀 Getting Started

### Prerequisites
* **Python 3.10.x**: It is highly recommended to use Python 3.10 to ensure library compatibility.
* **Webcam**: A functional camera for real-time inference.

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Atiqumer/SignSpeakAI
   cd SignSpeak-AI

### Set up a Virtual Environment:
2. python -m venv venv
# Windows:
.\venv\Scripts\activate

### Install Dependencies:
pip install -r requirements.txt