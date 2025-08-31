import os
import zipfile
import numpy as np
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sounddevice as sd

# =========================
# Step 1: Unzip dataset
# =========================
zip_path = "Emotion_1.zip"   # your dataset zip file
extract_path = "Emotion_1"   # extraction folder

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extracted dataset to:", extract_path)
else:
    print("✅ Dataset already extracted.")

# =========================
# Step 2: Feature extraction (fixed size)
# =========================
def extract_features(file, max_pad_len=130):  
    """Extract MFCC features from audio file with fixed length"""
    try:
        signal, sr = librosa.load(file, sr=22050)  # resample to 22.05 kHz
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

        # Pad or truncate to ensure same length
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("⚠️ Error processing", file, ":", e)
        return None

# =========================
# Step 3: Emotion label parser
# (Assumes RAVDESS-style filenames: 03-01-05-01-02-02-12.wav)
# =========================
def get_emotion_from_filename(file):
    try:
        parts = file.split("-")
        emotion_id = int(parts[2])  # 3rd number is emotion ID
        mapping = {
            1: "neutral",
            2: "calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgust",
            8: "surprised"
        }
        return mapping.get(emotion_id, "unknown")
    except:
        return "unknown"

# =========================
# Step 4: Load dataset
# =========================
X, y = [], []
for root, _, files in os.walk(extract_path):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            feature = extract_features(file_path)
            if feature is not None:
                X.append(feature)
                y.append(get_emotion_from_filename(file))

print("✅ Total valid samples loaded:", len(X))

# =========================
# Step 5: Train/Test split
# =========================
if len(X) > 0:
    X = np.array(X)  # convert to 2D array
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # Step 6: Train classifier
    # =========================
    model = MLPClassifier(hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model.fit(X_train, y_train)

    # =========================
    # Step 7: Evaluate
    # =========================
    y_pred = model.predict(X_test)
    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

    # =========================
    # Step 8: Real-time mic test
    # =========================
    def record_and_predict(duration=3, sr=22050):
        print("\n🎤 Recording... Speak now!")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()  # wait until recording finishes
        print("✅ Recording complete")

        # Convert to 1D array
        audio = audio.flatten()

        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfccs.shape[1] < 130:
            pad_width = 130 - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :130]

        features = np.mean(mfccs.T, axis=0).reshape(1, -1)

        # Predict
        prediction = model.predict(features)
        print("🎯 Detected Emotion:", prediction[0])

    # Run once after training
    record_and_predict(duration=3)

else:
    print("⚠️ No audio samples found! Check dataset structure.")
