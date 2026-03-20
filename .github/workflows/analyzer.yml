import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

def analyze_audio(file_path):
    try:
        # 1. Load the audio file (Force 22050Hz for consistency)
        y, sr = librosa.load(file_path, sr=22050)

        # 2. Compute Mel-scaled power spectrogram
        # n_mels=128 gives a good balance of detail for a dashboard
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        # 3. Convert to log scale (dB) - this is how humans "hear" volume
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 4. Generate the Visual Map (PNG)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Forensic Frequency Map')
        plt.tight_layout()
        
        output_image = "analysis_map.png"
        plt.savefig(output_image)
        plt.close()

        # 5. Export Raw Data for React (JSON)
        # We take a mean of the frequencies to simplify the dashboard load
        avg_freqs = np.mean(S_dB, axis=1).tolist()
        data = {
            "filename": os.path.basename(file_path),
            "sample_rate": sr,
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "spectral_data": avg_freqs
        }

        with open("analysis_data.json", "w") as f:
            json.dump(data, f)

        print(f"Success: Generated {output_image} and analysis_data.json")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Expects the filename as a command line argument from GitHub Actions
    if len(sys.argv) > 1:
        analyze_audio(sys.argv[1])
    else:
        print("No file provided.")
