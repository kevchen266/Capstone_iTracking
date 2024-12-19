import json
import base64
from random import sample
import cv2
import numpy as np
import requests  # 用于向API端点发送请求
from channels.generic.websocket import AsyncWebsocketConsumer
import logging
import os  # 用于文件操作
# from .db import insert_to_cdb
# insert_to_pdb, produce_heatmap
import io
import asyncio
# from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from .worker import start_calibration_workers
from .worker2 import start2
from .queue_manager import q as calibration_q
from .queue_manager import prediction_q as pred_q
from .queue_manager import model_output_q
from bson.binary import Binary
from .worker_prediction import start_prediction_workers
import base64
from PIL import Image
import io
from .worker_heatmap import start_heatmap_worker
from .db import produce_heatmap
from .queue_manager import heatmap_q, calibration
from .events import prediction_done
import time
from django.core.cache import cache
# from .mini_training import mini_train_step, model, optimizer, criterion, device

# DispatcherConsumer 中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_IMAGES_DIR = os.path.join(BASE_DIR, "../processed_images/prediction_receive")
CROPPED_IMAGES_DIR = os.path.join(BASE_DIR, "../processed_images/crop_image")


os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

logging.basicConfig(level=logging.WARNING)
# logger = logging.getLogger(__name__)
# Calibration Worker thread
# start()
# start2()

class DispatcherConsumer(AsyncWebsocketConsumer):
    def __init__(self):
        super().__init__()
        # self.calibration_phase = False
        # self.prediction_phase = False
        

    async def connect(self):
        """
        Handles the WebSocket connection establishment.
        - Accepts the WebSocket connection.
        - Starts prediction and heatmap workers immediately after connection.
        """
        # logger.debug("DispatcherConsumer: WebSocket 连接已建立")
        await self.accept()
        # calibration_phase = cache.get("calibration_phase", True)
        # prediction_phase = cache.get("prediction_phase", False)
        # print(f"Calibration Phase: {calibration_phase}, Prediction Phase: {prediction_phase}")
        # # start()
        # # self.calibration_phase = True
        # # self.prediction_phase = False
        # if calibration_phase and not prediction_phase:
        #     start_calibration_workers()
        #     # self.calibration_phase = False
        #     # self.prediction_phase = True
        #     cache.set("calibration_phase", False)
        #     cache.set("prediction_phase", True)
        # else:
        #     print("Prediction Phase Starting")
        #     # time.sleep(30)
        #     start_prediction_workers(num_workers=4)
        #     start_heatmap_worker()
        start_prediction_workers(num_workers=4)
        start_heatmap_worker()
    async def disconnect(self, close_code):
        """
        Handles WebSocket disconnection events.
        - Ensures heatmap queue (`heatmap_q`) is properly managed.
        - Prompts the user for input if the heatmap queue is empty.
        - Processes the heatmap queue if items exist.
        """
        # logger.debug(f"DispatcherConsumer: WebSocket 连接关闭")
        
        # if prediction_done.is_set():
        #     video_name = input("Enter which video name you want a heat map of: ")
        #     # frame_number = input("Enter frame number (For heatmap queue): ")
        #     heatmap_q.append(video_name)
        if not heatmap_q:
            # Prompt the user for video name and frame number
            video_name = input("Enter video name (For heatmap queue): ")
            # frame_number = input("Enter frame number (For heatmap queue): ")
            heatmap_q.append(video_name)
        else:
            for x in range(25):
                print("CARE")
            # Ensure proper structure of heatmap_q and process the first item
            if len(heatmap_q) > 0:
                video_name = heatmap_q.popleft()  # Correctly use popleft() for deque
                produce_heatmap(video_name)
            else:
                print("Heatmap queue structure is invalid.")
        
        # if self.calibration_phase:
            # start_calibration_workers()
            # self.calibration_phase = False
            # start_prediction_workers(num_workers=4)
           
            # start_heatmap_worker()
        
        
    async def receive(self, text_data="", bytes_data=None):
        # logger.debug(f"DispatcherConsumer 收到数据: {text_data if text_data else 'No text data'}")
        if bytes_data:
            
            print(f"Binary data received: (length: {len(bytes_data)})")  # 打印二進制數據長度

        try:
            if text_data.startswith("END"):
                prediction_done.set()
            if text_data.startswith("C"):
                # print(f"Calibration data keys: {list(json.loads(text_data[1:]).keys())}")  # 打印校準數據的 key
                await self.handle_calibration(text_data[1:])
                
            elif text_data.startswith("P"):
                prediction_data = json.loads(text_data[1:])
                # print(f"Prediction data keys: {list(prediction_data.keys())}")  # 打印預測數據的 key
                await self.handle_prediction(text_data[1:])
                
            elif text_data.startswith("RequestVideoURL"):
                print(f"Video request received, keys: RequestVideoURL")  # 固定鍵值
                request_data = text_data.split(':')
                if len(request_data) == 2 and request_data[0] == "RequestVideoURL":
                    try:
                        num_videos = int(request_data[1])
                        await self.handle_video_request(num_videos)
                    except ValueError:
                        await self.send(text_data=json.dumps({'error': '请求的视频数量无效'}))
                else:
                    await self.send(text_data=json.dumps({'error': '无效数据'}))
            else:
                print(f"Unknown message received, unable to determine keys.")  # 無法解析時提示
                await self.send(text_data=json.dumps({'error': '无效数据'}))
        except Exception as e:
            # logger.error(f"处理数据时出错")
            await self.send(text_data=json.dumps({'error': str(e)}))
            await self.close()
        

    async def handle_calibration(self, text_data):
        calibration_q.append(json.loads(text_data))
        print(json.loads(text_data)["coordinates"])
        print(f"Data added to calibration queue. Current size: {len(calibration_q)}")
        # if len(calibration_q) == 0 and not self.calibration_done:
        #     self.calibration_done = True  # 标记为已完成
        #     print("Calibration phase completed. Starting mini-training...")
        #     mini_train_step(model, optimizer, criterion, device, epochs=5)
        # base64_image = data["images"]
        # image_bytes = base64.b64decode(base64_image)

        # try:
        #     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # except Exception as e:
        #     # logger.warning(f"Image decoding failed, creating blank image: {e}")
        #     image = Image.new("RGB", (640, 360), color=(0, 0, 0))
        # print("receive data, the coordinates are:",data["coordinates"])
        # asyncio.create_task(self.icdb(data["calibration_spot"], bytes(data["images"])))
        
 
    async def handle_prediction(self, text_data):
        try:
            data = json.loads(text_data)
            base64_image = data["images"]
            relative_time = data["relativeTime"]
            video_index = data["videoIndex"]
            video_url = data["videos"][video_index]

            # Decode Base64 and convert to PIL image
            image_bytes = base64.b64decode(base64_image)
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                # logger.warning(f"Image decoding failed, creating blank image: {e}")
                image = Image.new("RGB", (640, 360), color=(0, 0, 0))

            # Save image locally and add to queue
            print("RELATIVE_TIME" + str(relative_time))
            frame_num = int((relative_time / 1000) * 30) + 1
            filename = f"{video_url.split('/')[-1]}_frame{frame_num}.jpg"
            save_path = os.path.join(PROCESSED_IMAGES_DIR, filename)
            image.save(save_path)

            meta_data = {"video_name": video_url, "frame_number": frame_num}
            pred_q.append([image, meta_data])
            # logger.info(f"Prediction data added to queue: {meta_data}")

        except Exception as e:
            # logger.error(f"Error handling prediction: {e}")
            await self.send(text_data=json.dumps({'error': 'Error handling prediction'}))

    async def handle_video_request(self, num_videos):


        all_video_urls = [
            "http://192.168.1.66:8000/videos/001_h264_1K.mp4",
            
        ]

        if num_videos > len(all_video_urls):
            num_videos = len(all_video_urls)

        selected_videos = sample(all_video_urls, num_videos)

        response = {
            'video_urls': selected_videos,
        }
        await self.send(text_data=json.dumps(response))

    
