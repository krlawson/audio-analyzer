import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import shutil

def analyze_audio(file_path):
    """Performs the spectral math and saves the visual/data maps."""
    # 1. Clean the filename for Git and Web compatibility
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Removes spaces, apostrophes, and dashes to prevent 'Exit 128'
    clean_name = base_name.replace(" ", "_").replace("'", "").replace("-", "_")
    
    print(f"--- Auditing: {clean_name} ---")
    
    # 2. Load and process audio (FFT Math)
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 3. Save the Physical Map (.png)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"map_{clean_name}.png") 
    plt.close()

    # 4. Save the Spectral Data (.json)
    spectral_data = np.mean(S_dB, axis=1).tolist()
    with open(f"data_{clean_name}.json", "w") as f:
        json.dump({"track": clean_name, "frequencies": spectral_data}, f)
    
    print(f"Success: Analysis complete for {clean_name}")

def generate_gallery():
    """Scans the repo for maps and builds the Forensic Gallery webpage."""
    maps = [f for f in os.listdir('.') if f.startswith('map_') and f.endswith('.png')]
    maps.sort()

    html_content = """
    <html>
    <head>
        <title>StudioGenius Forensic Gallery</title>
        <style>
            body { background: #0a0a0a; color: #00ff9d; font-family: 'Courier New', Courier, monospace; padding: 40px; }
            h1 { color: #00ff9d; border-bottom: 2px solid #00ff9d; padding-bottom: 10px; text-transform: uppercase; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 30px; margin-top: 30px; }
            .card { background: #1a1a1a; padding: 15px; border-radius: 4px; border: 1px solid #333; }
            .card:hover { border-color: #00ff9d; box-shadow: 0 0 10px #00ff9d; }
            img { width: 100%; border-radius: 2px; margin-top: 10px; filter: grayscale(20%); }
            .track-title { font-weight: bold; color: #ffffff; letter-spacing: 1px; }
            .meta { font-size: 10px; color: #666; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1>StudioGenius // Forensic Audit Gallery</h1>
        <div class="grid">
    """

    for m in maps:
        display_name = m.replace('map_', '').replace('.png', '').replace('_', ' ')
        html_content += f"""
            <div class="card">
                <div class="track-title">{display_name}</div>
                <div class="meta">SOURCE: {m}</div>
                <a href="{m}" target="_blank"><img src="{m}" /></a>
            </div>
        """

    html_content += "</div></body></html>"

    with open("index.html", "w") as f:
        f.write(html_content)
    print("Gallery updated: index.html is ready.")

if __name__ == "__main__":
    upload_folder = "uploads"
    processed_folder = "processed"
    
    # Ensure folders exist
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    if os.path.exists(upload_folder):
        # Catch wav, mp3, and m4a
        audio_files = [f for f in os.listdir(upload_folder) if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
        
        if not audio_files:
            print("Uploads folder is empty. Refreshing gallery from current assets...")
        
        for filename in audio_files:
            full_path = os.path.join(upload_folder, filename)
            
            # 1. RUN THE MATH
            try:
                analyze_audio(full_path)
                
                # 2. MOVE THE EVIDENCE (The Janitor step)
                # This clears the upload folder to stop the 'broken record' loop
                shutil.move(full_path, os.path.join(processed_folder, filename))
                print(f"Moved {filename} to processed storage.")
            except Exception as e:
                print(f"Forensic failure on {filename}: {e}")
    
    # 3. BUILD THE VIEW
    generate_gallery()


