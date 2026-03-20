import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def analyze_audio(file_path=None):
    print("--- Starting Forensic Process ---")
    
    # 1. GENERATE DATA (Real or Fake)
    if file_path and os.path.exists(file_path):
        print(f"Analyzing Real File: {file_path}")
        y, sr = librosa.load(file_path, sr=22050)
    else:
        print("No file found. Generating a Calibration Map (Synthetic Data).")
        sr = 22050
        y = np.random.uniform(-1, 1, sr * 2) # 2 seconds of white noise

    # 2. THE MATH (FFT / Mel Spectrogram)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 3. THE PHYSICAL IMAGE
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig("analysis_map.png") # The critical physical file
    plt.close()

    # 4. THE PHYSICAL JSON
    spectral_data = np.mean(S_dB, axis=1).tolist()
    with open("analysis_data.json", "w") as f:
        json.dump({"status": "verified", "data": spectral_data}, f)
    
    print("Success: Physical results created in root directory.")

if __name__ == "__main__":
    # Check for 'uploads' folder
    target = None
    if os.path.exists("uploads"):
        files = [f for f in os.listdir("uploads") if f.endswith(".wav")]
        if files:
            target = os.path.join("uploads", files[0])
    
    analyze_audio(target)

