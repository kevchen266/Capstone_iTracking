from .queue_manager import model_output_q
from .db import insert_to_pdb
import cv2
import io
import time
import threading
import logging
import os

logger = logging.getLogger(__name__)

def process_prediction():
    while True:
        if model_output_q:
            try:
                # Dequeue a frame
                data = model_output_q.pop()
                video_name = data[0].split("/")[-1]  # Extract the file name from the URL
                video_path = os.path.join("videos", video_name)  # Build the full local path
                # Check if the video file exists
                if not os.path.exists(video_path):
                    logger.error(f"Video file does not exist: {video_path}")
                    continue
                # Open the video and set the frame position
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Failed to open video file: {video_path}")
                    continue
                frame_number = int(data[1])  # Ensure the frame number is an integer
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_number >= total_frames:
                    logger.error(f"Frame number {frame_number} exceeds total frames {total_frames} in video {video_path}")
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = cap.read()
                cap.release()
                if not success or frame is None:
                    logger.error(f"Failed to read frame {frame_number} from video {video_path}")
                    continue
                # Encode the frame
                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    logger.error(f"Failed to encode frame {frame_number} to JPG")
                    continue
                frame_binary = io.BytesIO(buffer).getvalue()
                # Insert into database
                insert_to_pdb(data[0], frame_number, frame_binary, data[2], data[3])

            except ValueError as ve:
                logger.error(f"ValueError: {ve} - Ensure data[1] is an integer")
            except Exception as e:
                logger.error(f"ERROR WRITING TO DB: {e}")
        else:
            time.sleep(0.1)  # Avoid busy-waiting when the queue is empty

def start_heatmap_worker():
    threading.Thread(target=process_prediction, daemon=True).start()

