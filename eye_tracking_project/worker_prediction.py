import threading
import time
import cv2
import mediapipe as mp
import logging
import torch
from torchvision import transforms
import os
import csv
from .queue_manager import prediction_q, model_output_q
from .CNN_model.Gazetrack import Gazetrack
import numpy as np
from .worker_heatmap import process_prediction
logger = logging.getLogger(__name__)

# Initialize the CNN model
model = Gazetrack()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CNN_model/cnn_model_weights.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define output directory
output_dir = os.path.join(BASE_DIR, "../processed_images/output")
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, "predictions.csv")

# Initialize the CSV file
def initialize_output_csv():
    """Initialize the CSV file with headers."""
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Video Name", "Frame Number", "Gaze X", "Gaze Y"])
            # logger.info(f"Initialized output CSV at {output_csv_path}")

initialize_output_csv()

def process_prediction():
    """Process images from the queue, crop eye regions, predict gaze, save results, and output to CSV."""
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    processed_images_dir = os.path.join(BASE_DIR, "../processed_images/crop_image")
    os.makedirs(processed_images_dir, exist_ok=True)

    while True:
        if prediction_q:
            try:
                # Retrieve image and metadata from the queue
                image, metadata = prediction_q.popleft()
                rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Detect facial landmarks
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye, right_eye = crop_eyes(rgb_image, face_landmarks)
                        if left_eye is not None and right_eye is not None:
                            save_cropped_images(left_eye, right_eye, metadata, processed_images_dir)
                            result = send_to_cnn_model(left_eye, right_eye)
                            if result:
                                store_prediction(metadata, result)                                
                                append_prediction_to_csv(metadata, result)
                                
                        else:
                            # logger.warning(f"Failed to crop eyes for frame {metadata['frame_number']}")
                            print()
                else:
                    # logger.warning(f"No facial landmarks detected for frame {metadata['frame_number']}")
                    print()
            except Exception as e:
                # logger.error(f"Error processing prediction: {e}")
                print()
        else:
            time.sleep(0.1)

def save_cropped_images(left_eye, right_eye, metadata, save_dir):
    """Save cropped images to the local directory."""
    try:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Retrieve metadata
        video_name = metadata["video_name"].split("/")[-1]  # Extract the video file name
        frame_number = metadata["frame_number"]

        # Generate file paths
        left_eye_filename = os.path.join(save_dir, f"{video_name}_frame{frame_number}_left_eye.jpg")
        right_eye_filename = os.path.join(save_dir, f"{video_name}_frame{frame_number}_right_eye.jpg")

        # Save images
        if left_eye is not None:
            cv2.imwrite(left_eye_filename, cv2.cvtColor(left_eye, cv2.COLOR_RGB2BGR))
        if right_eye is not None:
            cv2.imwrite(right_eye_filename, cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR))

        # logger.info(f"Cropped images saved for frame {frame_number}")
    except Exception as e:
        # logger.error(f"Error saving cropped images: {e}")
        print()

def crop_eyes(image, face_landmarks, expand_ratio=0.5, target_size=224):
    """Crop left and right eye images with bounding box expansion and resizing."""
    img_height, img_width, _ = image.shape
    left_eye_indices = [33, 133, 160, 158, 153, 144, 145, 161]
    right_eye_indices = [362, 263, 387, 385, 373, 380, 374, 386]

    def get_expanded_bbox(indices):
        x_coords = [int(face_landmarks.landmark[i].x * img_width) for i in indices]
        y_coords = [int(face_landmarks.landmark[i].y * img_height) for i in indices]
        if not x_coords or not y_coords:
            return None
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        max_side = max(x_max - x_min, y_max - y_min) * (1 + expand_ratio)
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
        x_min, x_max = max(0, x_center - int(max_side // 2)), min(img_width, x_center + int(max_side // 2))
        y_min, y_max = max(0, y_center - int(max_side // 2)), min(img_height, y_center + int(max_side // 2))
        return x_min, x_max, y_min, y_max

    def extract_eye_region(indices):
        bbox = get_expanded_bbox(indices)
        if bbox:
            x_min, x_max, y_min, y_max = bbox
            eye_image = image[y_min:y_max, x_min:x_max]
            if eye_image.size > 0:
                return cv2.resize(eye_image, (target_size, target_size))
        return None

    left_eye = extract_eye_region(left_eye_indices)
    right_eye = extract_eye_region(right_eye_indices)
    return left_eye, right_eye

def send_to_cnn_model(left_eye, right_eye):
    """Send cropped eye images to the CNN model and return the prediction."""
    try:
        left_eye_tensor = transform(left_eye).unsqueeze(0)
        right_eye_tensor = transform(right_eye).unsqueeze(0)
        with torch.no_grad():
            output = model(left_eye_tensor, right_eye_tensor)
        return {"gaze_coordinates": output.cpu().numpy().flatten()}
    except Exception as e:
        # logger.error(f"Error during CNN model prediction: {e}")
        return None

def store_prediction(metadata, result):
    """Store prediction results in the output queue."""
    try:
        video_name = metadata["video_name"]
        frame_number = metadata["frame_number"]
        x, y = result["gaze_coordinates"]
        model_output_q.append([video_name, str(frame_number), f"{x:.4f}", f"{y:.4f}"])
        # logger.info(f"Prediction stored for frame {frame_number}")
    except Exception as e:
        # logger.error(f"Error storing prediction: {e}")
        print()

def append_prediction_to_csv(metadata, result):
    """Append prediction results to the output CSV file."""
    try:
        video_name = metadata["video_name"]
        frame_number = metadata["frame_number"]
        x, y = result["gaze_coordinates"]
        with open(output_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([video_name, frame_number, f"{x:.4f}", f"{y:.4f}"])
        # logger.info(f"Prediction for frame {frame_number} written to CSV")
    except Exception as e:
        # logger.error(f"Error appending prediction to CSV: {e}")
        print()

def start_prediction_workers(num_workers=4):
    """Start worker threads for prediction processing."""
    for _ in range(num_workers):
        threading.Thread(target=process_prediction, daemon=True).start()
