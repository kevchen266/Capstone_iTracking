import asyncio
import websockets
import json
import base64
import requests
import os

def download_image(url, save_path):
    '''Download images from a URL and save locally'''
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved to {save_path}")
    else:
        print(f"Failed to download image from {url}, status code: {response.status_code}")

def download_test_images():
    '''Download multiple test images to simulate real data'''
    save_directory = "./images"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(30):
        image_url = f"https://randomuser.me/api/portraits/men/{i % 100}.jpg"
        save_path = os.path.join(save_directory, f"image_{i+1}.jpg")
        download_image(image_url, save_path)

async def test_websocket_prediction():
    uri = "ws://localhost:8000/ws/dispatcher/"
    save_directory = "./images"

    for i in range(1, 31):
        try:
            async with websockets.connect(uri) as websocket:
                print("WebSocket connected")

                # 获取图片路径
                image_path = os.path.join(save_directory, f"image_{i}.jpg")

                # 加载并编码图像
                with open(image_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8').strip()

                
                # 构建 JSON 数据
                prediction_data = {
                    "images": [base64_image],  # 确保 images 是一个列表
                    "videos": ["video1.mp4", "video2.mp4"],
                    "relativeTime": 500 + i * 10,
                    "videoIndex": i % 2
                }
                json_data = "P" + json.dumps(prediction_data)  # 添加 "P" 前缀


                # 发送 JSON 数据
                print(f"Sending prediction data, image #{i}")
                await websocket.send(json_data)

                # 接收响应
                response = await websocket.recv()
                print(f"Received response: {response}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"WebSocket connection closed: {e}")
            await asyncio.sleep(1)  # 等待重试

# 下载图片
download_test_images()

# 启动预测测试
asyncio.run(test_websocket_prediction())
