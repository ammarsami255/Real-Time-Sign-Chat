# Real-Time Sign Chat

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue)

## 📌 Overview
Real-Time Sign Chat is a Python-based application that provides:
- **Sign Language Recognition** using MediaPipe and TensorFlow Lite.
- **Voice Recognition** converting speech to text and displaying corresponding images.

## 🚀 Features
✅ Real-time hand gesture recognition for sign language.
✅ Converts recognized signs into text.
✅ Speech-to-text conversion using Google Speech Recognition.
✅ Displays images corresponding to recognized words.
✅ User-friendly GUI built with `customtkinter`.

## 📷 Demo
![App Screenshot](https://via.placeholder.com/600x300.png?text=Real-Time+Sign+Chat+Demo)

## 🛠️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-sign-chat.git
cd real-time-sign-chat
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
python gui.py
```

## 📂 Project Structure
```
real-time-sign-chat/
│── gui.py             # GUI with CustomTkinter
│── main.py            # Main logic for sign & voice recognition
│── model.tflite       # Trained TensorFlow Lite model
│── requirements.txt   # Required dependencies
│── README.md          # Project Documentation
└── data/              # Image dataset for letters & numbers
```

## 🏗️ Technologies Used
- **Python** (Core language)
- **TensorFlow Lite** (Sign language recognition model)
- **MediaPipe** (Hand tracking)
- **OpenCV** (Image processing)
- **SpeechRecognition** (Voice input)
- **CustomTkinter** (GUI design)

## 💡 Usage
1. Click **"Sign Language Recognition"** to start sign detection.
2. Click **"Voice Recognition"** to convert speech to sign images.
3. Press **'Q'** or **Esc** to exit the sign recognition mode.


