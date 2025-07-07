# Dyxlexia-at-ease

# ✨ Dyslexia Learning Web App – *DyslexiaAtEase*

A simple and interactive web app designed to help children with dyslexia practice **letter tracing** and **spelling pronunciation** using **machine learning** and **speech recognition**. This tool aims to provide a fun, guided experience to support foundational language skills development.

![WhatsApp Image 2025-05-18 at 22 43 14_f6772daf](https://github.com/user-attachments/assets/8fb5e34a-0a0d-40ef-9861-38730c1560dd)


## 🧠 Key Features

- ✍️ **Letter Tracing Practice**
  - Faded letter image guides children to trace the alphabet.
  - Real-time feedback based on similarity index (e.g., 80% match required to proceed).
  - Automatically progresses to the next letter on successful tracing.
  - Tracks individual child’s progress.

- 🧪 **Letter Recognition**
  - Uses a pre-trained CNN model (EMNIST) to analyze the user-drawn letter.
  - Compares it with the expected letter and calculates similarity using cosine similarity.

- 🔊 **Spelling Practice with Voice Feedback**
  - Built-in text-to-speech reads out a word.
  - User repeats the word; app uses Vosk for speech-to-text.
  - Calculates pronunciation accuracy and gives hints if needed.

- 📊 **Child Progress Dashboard**
  - Shows completed letters and performance.
  - Ideal for parents or teachers to track each child.

---

## 🖼️ Screenshots

| Letter Writing Practice | Voice Pronunciation Feedback |
|--------------------------|------------------------------|
| ![Letter Writing](images/letter_practice.png) | ![Voice Feedback](images/voice_feedback.png) |

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript, Canvas API
- **Backend:** Flask (Python)
- **Machine Learning:** TensorFlow/Keras (EMNIST character model)
- **Speech Recognition:** Vosk
- **Text-to-Speech:** pyttsx3
- **Database (optional):** MySQL / Hardcoded users (for prototype)

---

## 🚀 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/dyslexiaat-ease.git
   cd dyslexiaat-ease

