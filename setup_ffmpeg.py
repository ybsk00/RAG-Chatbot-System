import os
import zipfile
import urllib.request
import sys
import shutil

def setup_ffmpeg():
    print("Downloading FFmpeg (Lightweight version)...")
    # Download URL for a lightweight static build of ffmpeg for Windows
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = "ffmpeg.zip"
    
    try:
        # Download
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("ffmpeg_temp")
            
        # Find ffmpeg.exe
        ffmpeg_exe = None
        for root, dirs, files in os.walk("ffmpeg_temp"):
            if "ffmpeg.exe" in files:
                ffmpeg_exe = os.path.join(root, "ffmpeg.exe")
                break
        
        if ffmpeg_exe:
            # Move to current directory
            target_path = os.path.join(os.getcwd(), "ffmpeg.exe")
            shutil.move(ffmpeg_exe, target_path)
            print(f"Success! ffmpeg.exe is ready at: {target_path}")
            
            # Clean up
            os.remove(zip_path)
            shutil.rmtree("ffmpeg_temp")
            
            # Verify
            print("Verifying installation...")
            import subprocess
            subprocess.run([target_path, "-version"])
        else:
            print("Error: Could not find ffmpeg.exe in the downloaded zip.")
            
    except Exception as e:
        print(f"Error setting up ffmpeg: {e}")

if __name__ == "__main__":
    setup_ffmpeg()
