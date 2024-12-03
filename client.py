import asyncio
import websockets
import json
import base64
import cv2

async def simulate_websocket():
    uri = "ws://localhost:8000/ws/dispatcher/"

    async with websockets.connect(uri) as websocket:
        # Example 1: Send calibration data
        # calibration_image = cv2.imread("path/to/calibration_image.jpg")
        # _, buffer = cv2.imencode('.jpg', calibration_image)
        # calibration_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # calibration_data = {
        #     'image': f'data:image/jpeg;base64,{calibration_image_base64}',
        #     'calibration_spot': [100, 200]
        # }
        # await websocket.send('C' + json.dumps(calibration_data))
        # response = await websocket.recv()
        # print("Calibration Response:", response)

        # Example 3: Send video request data
        video_request_data = "RequestVideoURL:2"
        await websocket.send(video_request_data)
        response = await websocket.recv()
        print("Video Request Response:", response)
        video_urls = json.loads(response).get("video_urls", [])
        print(video_urls)
        
        # Example 2: Send image prediction data
        # prediction_image = cv2.imread("path/to/prediction_image.jpg")
        # _, buffer = cv2.imencode('.jpg', prediction_image)
        # prediction_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        prediction_data = {
            # 'image': f'data:image/jpeg;base64,{prediction_image_base64}',
            "image": "image_1",
            "videos": video_urls,
            "video": 1,
            "r_time": 5 # relative time in seconds - e.g. 00:02:33.33 -> 153.33
        }
        await websocket.send("P" + json.dumps(prediction_data))
        response = await websocket.recv()
        print("Prediction Response:", response)

# Run the simulation
asyncio.get_event_loop().run_until_complete(simulate_websocket())
