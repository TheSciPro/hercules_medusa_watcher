import cv2
import os

# Load video
video_path = "/data/shared/users/antara/rag/video/media/video.webm"
output_dir = "/data/shared/users/antara/rag/video/output/frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 3)  

frame_count = 0
saved_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        timestamp_sec = int(frame_count / fps)
        frame_filename = f"frame_{timestamp_sec}s.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        saved_frames.append(frame_path)

    frame_count += 1

cap.release()

saved_frames[:5]  # Show first few saved frame paths for confirmation
