print("WORKER FILE")

import threading
import time
from .queue_manager import q as calibration_q
from .queue_manager import dataloader_q 
import logging
from .db import insert_to_cdb
import base64
import mediapipe as mp
import os
import cv2
import numpy as np
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .CNN_model.Gazetrack import Gazetrack
import torch
from PIL import Image
import io
from .mini_training import mini_train_step, model, optimizer, criterion, device
import sys
# logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_IMAGES_DIR = os.path.join(BASE_DIR, "../processed_images/calibration_cropped_images")
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Gazetrack().to(device)

# Paths for weights
default_model_path = os.path.join(BASE_DIR, "CNN_model/cnn_model_weights.pth")
model.eval()



def crop_eyes(image, face_landmarks, coordinates,  expand_ratio=0.5, target_size=224):
    """Crop left and right eye images with bounding box expansion and resizing."""
    print("X, Y INSIDE CROPPED EYES FUNCTION", coordinates)
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
                resized_eye = cv2.resize(eye_image, (target_size, target_size))
                # # tensor_eye = torch.tensor(resized_eye, dtype=torch.float32).permute(2, 0, 1).contiguous()
                # # print(f"Extracted eye region: {bbox}, Tensor shape: {tensor_eye.shape}")
                # return resized_eye, tensor_eye
                return cv2.cvtColor(resized_eye, cv2.COLOR_BGR2RGB)
        print(f"Failed to extract eye region: indices={indices}, bbox={bbox}")
        # return None, tensor_eye
        return None

    left_eye = extract_eye_region(left_eye_indices)
    right_eye = extract_eye_region(right_eye_indices)
    print(f"Left eye extracted successfully: {type(left_eye[0])}, {type(left_eye[1])}")
    print(f"Right eye extracted successfully: {type(right_eye[0])}, {type(right_eye[1])}")
    return left_eye, right_eye, coordinates



def save_cropped_images(left_eye, right_eye, coordinates, save_dir):
    """Save cropped images to the local directory."""
    try:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique identifiers (e.g., using timestamps)
        timestamp = int(time.time() * 1000)  # Milliseconds since epoch

        # Generate file paths with unique names
        left_eye_filename = os.path.join(save_dir, f"{coordinates}_left_eye_{timestamp}.jpg")
        right_eye_filename = os.path.join(save_dir, f"{coordinates}_right_eye_{timestamp}.jpg")

        # Save images
        if left_eye is not None:
            cv2.imwrite(left_eye_filename, cv2.cvtColor(left_eye, cv2.COLOR_RGB2BGR))
        if right_eye is not None:
            cv2.imwrite(right_eye_filename, cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR))

        print(f"Images saved: {left_eye_filename}, {right_eye_filename}")
        folder_path = "/Users/kev19/Desktop/Project/summer project/eyestracking/full-stack/Eyes_tracking 2/processed_images/calibration_cropped_images"
        items = os.listdir(folder_path)
        item_count = len(items)
        print(f"The folder contains {item_count} items.")
        # if isinstance(left_eye, np.ndarray):
        #     left_eye_filename = os.path.join(save_dir, f"{coordinates}_left_eye_{timestamp}.jpg")
        #     cv2.imwrite(left_eye_filename, cv2.cvtColor(left_eye, cv2.COLOR_RGB2BGR))
        #     print(f"Left eye saved at: {left_eye_filename}")

        # if isinstance(right_eye, np.ndarray):
        #     right_eye_filename = os.path.join(save_dir, f"{coordinates}_right_eye_{timestamp}.jpg")
        #     cv2.imwrite(right_eye_filename, cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR))
        #     print(f"Right eye saved at: {right_eye_filename}")

    except Exception as e:
        print(f"Error saving cropped images: {e}")


def process_calibration():
    """处理校准帧并执行小规模训练。
       * 归一化后的 gaze 坐标会作为标签
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    a=len(calibration_q)
    # print(a)
    while len(calibration_q) < 90:
        continue
    print("CALIBRATION _Q is READY")
    a=len(calibration_q)
    while calibration_q:
        if calibration_q:
            try:
                data = calibration_q.popleft()
                base64_image = data["images"]
                image_bytes = base64.b64decode(base64_image)
                coordinates = data["coordinates"]
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    # logger.warning(f"Image decoding failed, creating blank image: {e}")
                    image = Image.new("RGB", (640, 360), color=(0, 0, 0))
                print("receive data, the coordinates are:",data["coordinates"])

            
                rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # 对坐标进行归一化处理
                screen_width, screen_height = 1600, 720
                normalized_coordinates = [
                    coordinates[0] / screen_width,
                    coordinates[1] / screen_height
                ]

                # 检测面部关键点
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # print(f"偵測成功：{face_landmarks.landmark}")
                        # print("X, Y BEFORE (OUTSIDE) CROPPED EYES FUNCTION CALLED", coordinates)
                        left_eye, right_eye , cropped_coordinates = crop_eyes(rgb_image, face_landmarks, normalized_coordinates)

                        """
                        [left_eye_image,right_eye_image, label] ->  dataloader_q (global) ->(seperate file call when disconnected) loop through queue to load into dataloader ->
                         -> model -> save new weigt.pth

                         original data-> image -> image-> crop_eyes() -> left_cropped, right_cropped -> dataloader_q
                                       ->   coordinates -> dataloaderq                  
                        """
                        if left_eye is not None and  right_eye is not None and cropped_coordinates is not None:
                            # 保存到本地
                            
                            save_cropped_images(left_eye, right_eye, coordinates, PROCESSED_IMAGES_DIR)

                            # # 用張量進行模型訓練
                            # left_eyes.append(left_eye)
                            # right_eyes.append(right_eye)
                            # labels.append(normalized_coordinates)
                            dataloader_q.append([left_eye,right_eye, cropped_coordinates])

                            # 如果緩衝區已滿，觸發小規模訓練
                            # if len(left_eyes) >= 32:
                            #     left_tensor = torch.stack(left_eyes).to(device)
                            #     right_tensor = torch.stack(right_eyes).to(device)
                            #     labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

                            #     mini_train_step(model, optimizer, criterion, left_tensor, right_tensor, labels_tensor, device)

                            #     left_eyes.clear()
                            #     right_eyes.clear()
                            #     labels.clear()
            except Exception as e:
                print(f"校准处理时出错: {e}")
    else:
        # time.sleep(0.1)
        print("CALIBRATION_Q is EMPTY")
        if a ==len(dataloader_q):
            print("GOOD")
        print(BASE_DIR)
        mini_train_step(model, optimizer, criterion, device, epochs=1)
        



# def start():
#     threading.Thread(target=process_calibration, daemon=True).start()

def start_calibration_workers(num_workers=1):
    """Start worker threads for prediction processing."""
    for _ in range(num_workers):
        threading.Thread(target=process_calibration, daemon=True).start()