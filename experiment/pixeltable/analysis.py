import os
import pytesseract
from PIL import Image

# Folder where your frames are saved
frames_dir = "output/frames"

ocr_results = {}
for frame_file in sorted(os.listdir(frames_dir)):
    if frame_file.endswith(".jpg"):
        path = os.path.join(frames_dir, frame_file)
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        ocr_results[frame_file] = text.strip()

# Print or save results
for frame, text in ocr_results.items():
    print(f"{frame}: {text}")
