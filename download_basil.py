import os
import urllib.request
import zipfile
from pathlib import Path

# URL for Basil Health Detection dataset (YOLOv8 format)
# This is a direct link that works without API key
url = "https://universe.roboflow.com/api/dataset/basil-health-detection-9crhs/download/yolov8"

print("📥 Downloading Basil Health Detection dataset...")
print(f"   URL: {url}")

try:
    # Download
    zip_path = "basil_dataset_temp.zip"
    urllib.request.urlretrieve(url, zip_path)
    print(f"✅ Downloaded: {zip_path}")
    
    # Extract
    extract_dir = "basil_dataset"
    if os.path.exists(extract_dir):
        import shutil
        shutil.rmtree(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"✅ Extracted to: {extract_dir}")
    
    # Cleanup
    os.remove(zip_path)
    
    # Show structure
    print("\n📊 Dataset structure:")
    os.system(f"ls -la {extract_dir}/ && echo '---' && ls -la {extract_dir}/images/ 2>/dev/null | head -5")
    
except Exception as e:
    print(f"⚠️  Error: {e}")
    print("\nManual download required from:")
    print("https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs")

