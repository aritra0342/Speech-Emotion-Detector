# 🎤 Speech Emotion Detector  

A Python-based **real-time Speech Emotion Recognition** project that listens to your voice through your microphone and predicts your current emotion (e.g., happy, sad, angry, etc.) using **MFCC features** and an **MLP classifier**.

---

###  GitHub Repo

The full project is hosted here: [speech-emotion-detector](https://github.com/your-username/speech-emotion-detector)

---

##  Dataset

This project uses the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, a validated multimodal collection of emotional speech and song performed by 24 professional actors across several emotions (calm, happy, sad, angry, fearful, surprise, disgust, neutral). Each expression is available in multiple modalities at different intensities and can be downloaded under a Creative Commons license from [Zenodo](https://zenodo.org/records/1188976) :contentReference[oaicite:0]{index=0}.

---

##  Features
-  Trains on the RAVDESS dataset (speech-only, Emotion_1.zip)
-  Extracts **MFCC audio features** (standardized to fixed-length)
-  Achieves ~78% accuracy with an **MLP classifier**
-  Real-time emotion prediction from your **microphone input**
-  Built with **Python**, **Librosa**, **scikit-learn**, **sounddevice**, etc.

---

##  Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/speech-emotion-detector.git
cd speech-emotion-detector
