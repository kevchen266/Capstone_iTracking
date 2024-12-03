import json
import base64
import random
import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import logging
import os

import requests

logger = logging.getLogger(__name__)

class TestDispatcherConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        logger.debug("DispatcherConsumer: WebSocket 连接已建立")
        await self.accept()

    async def disconnect(self, close_code):
        logger.debug(f"DispatcherConsumer: WebSocket 连接关闭: {close_code}")

    async def receive(self, text_data=None, bytes_data=None):
        logger.debug(f"DispatcherConsumer 收到数据")
        try:
            if bytes_data:  # 当接收到字节数据时
                text_data = bytes_data.decode('utf-8')

            if text_data:
                # 检查消息的前缀以区分不同类型的请求
                if text_data.startswith('C'):
                    json_data = text_data[1:]  # 去掉 'C' 前缀，解析 JSON 数据
                    data = json.loads(json_data)
                    await self.handle_calibration(data)
                elif text_data.startswith('P'):
                    json_data = text_data[1:]  # 去掉 'P' 前缀，解析 JSON 数据
                    data = json.loads(json_data)
                    await self.handle_prediction(data)
                elif text_data.startswith('RequestVideoURL:'):
                    logger.debug(f"RequestVideoURL received, processing...:{text_data}")
                    # 获取视频数量
                    num_videos = int(text_data.split(':')[1])
                    logger.debug(f"RequestVideoURL received, processing...:{text_data}{type(text_data)}")
                    await self.handle_video_request(num_videos)
                else:
                    await self.send(text_data=json.dumps({'error': '无效数据'}))
        except Exception as e:
            logger.error(f"处理数据时出错: {e}")
            await self.send(text_data=json.dumps({'error': str(e)}))
            await self.close()

    async def handle_calibration(self, data):
        logger.debug("处理校准数据")
        await self.process_image(data, 'calibration')

    async def handle_prediction(self, data):
        logger.debug("处理预测数据")
        await self.process_image(data, 'prediction')

    async def process_image(self, data, stage):
        try:
            # 从 JSON 数据中提取图像的 Base64 编码数据
            image_data_base64 = data['images']
            image_bytes = base64.b64decode(image_data_base64)
            np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("图像解码失败")

            # 预测阶段不包含坐标信息
            if stage == 'calibration':
                coordinates = data.get('coordinates', [0, 0])
                file_name = f'{stage}_{coordinates[0]}_{coordinates[1]}.jpg'
            else:
                file_name = f'{stage}.jpg'

            # 处理图像并裁剪眼睛区域
            cropped_image, _ = self.detect_and_crop_eye_pair(image)

            # 保存裁剪后的图像
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_directory = os.path.join(base_dir, 'processed_images', stage)
            os.makedirs(save_directory, exist_ok=True)
            file_path = os.path.join(save_directory, file_name)
            cv2.imwrite(file_path, cropped_image)
            logger.debug(f"图像已保存为: {file_path}")

        except KeyError as e:
            logger.error(f"关键字段缺失: {e}")
            await self.send(text_data=json.dumps({'error': f'关键字段缺失: {str(e)}'}))
        except Exception as e:
            logger.error(f"处理图像数据时出错: {e}")
            await self.send(text_data=json.dumps({'error': str(e)}))

    def detect_and_crop_eye_pair(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        eye_pair_cascade_path = 'cascades/haarcascade_mcs_eyepair_big.xml'
        eye_pair_cascade = cv2.CascadeClassifier(eye_pair_cascade_path)

        eyes = eye_pair_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(20, 20)
        )
        coordinates = []
        cropped_image = image

        for (x, y, w, h) in eyes:
            coordinates.append((x, y, x + w, y, x, y + h, x + w, y + h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped_image = image[y:y + h, x:x + w]

        return cropped_image, coordinates
    
    async def handle_video_request(self, num_videos):
        logger.debug(f"Received num_videos: {num_videos}, Type: {type(num_videos)}")
        # 模拟获取视频 URL 的方法
        all_video_urls = [
            "http://192.168.1.144:8000/videos/001_h264_1K.mp4",
            "http://192.168.1.144:8000/videos/002_h264_1K.mp4",
          
            
            # 你可以继续添加更多视频URL
        ]

        if num_videos > len(all_video_urls):
            num_videos = len(all_video_urls)

        selected_videos = random.sample(all_video_urls, num_videos)

        response_string = "VideoURLs:" + ",".join(selected_videos)

        # Log the selected URLs to the terminal
        logger.debug(f"发送到前端的视频 URL: {response_string}")

        await self.send(text_data=response_string)