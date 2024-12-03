print("DB FILE")

from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging

logging.getLogger("pymongo").setLevel(logging.WARNING)

uri = 'mongodb+srv://eye-gaze-db-user:XEWUsuxRlPjdFcdi@eye-gaze-cluster.rb0cn.mongodb.net/?retryWrites=true&w=majority&appName=eye-gaze-cluster'

client = MongoClient(uri, serverSelectionTimeoutMS=20000)

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)



db = client["eye-gaze-db"]
# create a new db and/or collection if not already exist
cali_collection = client["eye-gaze-db"]["calibration"]
predict_collection = client["eye-gaze-db"]["prediction"]
fs = gridfs.GridFS(db)

def calibration_collection():
    return cali_collection

# new_spot = {
#         "calibration_spot": "ok",
#         "chunk": 1,
#         "images": []
#     }
# cali_collection.insert_one(new_spot)

def insert_to_cdb(calibration_spot, image):
    '''Insert calibration data'''

    image_id = fs.put(image, filename="cali_image.jpg")

    try:
        new_document = {
            "calibration_spot": calibration_spot,
            "image_id": image_id
        }
        cali_collection.insert_one(new_document)
    except Exception as e:
        print(f"Error inserting frame {calibration_spot} into DB: {e}")



    # existing_spot = cali_collection.find_one({"calibration_spot": calibration_spot}, sort=[("chunk", -1)])

    # if existing_spot:
    #     current_chunk = existing_spot["chunk"]
    #     if len(existing_spot["images"]) < 1000:
    #         cali_collection.update_one({"_id": existing_spot["_id"]}, {"$push": {
    #             "images": image
    #         }})
    #     else:
    #         updated_chunk = current_chunk + 1
    #         new_document = {
    #             "calibration_spot": calibration_spot,
    #             "chunk": updated_chunk,
    #             "images": [image]
    #         }
    #         cali_collection.insert_one(new_document)
    # else:
    #     new_spot = {
    #         "calibration_spot": calibration_spot,
    #         "chunk": 1,
    #         "images": [image]
    #     }
    #     cali_collection.insert_one(new_spot)

def insert_to_pdb(video_name, frame, frame_binary, x, y):
    '''Insert a session'''

    frame_id = fs.put(frame_binary, filename=video_name + "_frame_" + str(frame) + ".jpg")
    
    print(video_name)
    print(frame)

    existing_video = predict_collection.find_one({"video_name": video_name, "frame": frame}, sort=[("chunk", -1)])

    if existing_video:
        current_chunk = existing_video["chunk"]
        if len(existing_video["coordinates"]) < 1000:
            predict_collection.update_one({"_id": existing_video["_id"]}, {"$push": {
                "coordinates": [x, y]
            }})
        else:
            updated_chunk = current_chunk + 1
            new_document = {
                "video_name": video_name,
                "frame": frame,
                "frame_id": frame_id,
                "chunk": updated_chunk,
                "coordinates": [[x, y]]
            }
            predict_collection.insert_one(new_document)
    else:
        new_video = {
            "video_name": video_name,
            "frame": frame,
            "frame_id": frame_id,
            "chunk": 1,
            "coordinates": [[x, y]]
        }
        predict_collection.insert_one(new_video)

def produce_heatmap(video_name, frame):
    coordinates = []
    documents = predict_collection.find({"video_name": video_name, "frame": frame}, {"frame_id": 1, "coordinates": 1, "_id": 0})
    for doc in documents:
        frame_id = doc["frame_id"]
        for coordinate in doc['coordinates']:
            coordinates.append(coordinate)

    # data = {
    #     'x': [coordinate[0] for coordinate in coordinates],
    #     'y': [coordinate[1] for coordinate in coordinates]
    # }

    print(coordinates)
    # print(data)

    grid_out = fs.get(frame_id)
    frame = Image.open(io.BytesIO(grid_out.read()))
    image_width, image_height = frame.width, frame.height # Should be the same, I believe?

    data = {
        'x': np.random.randint(700, 1700 + 1, 100),
        'y': np.random.randint(500, 1100 + 1, 100)
    }

    df = pd.DataFrame(data)
    # df = df[((df['x'] >= 800) & (df['x'] <= 1000) & (df['y'] <= 900)) | ((df['x'] >= 1001) & (df['x'] <= 1200) & (df['y'] <= 900))]
    print(df)

    heatmap = np.zeros((image_height, image_width)) # Grid layout of a blank heatmap

    # Populating the heatmap
    intensity = 100
    for x, y in zip(df['x'], df['y']):
        x_int = int(x)
        y_int = int(y)
        if 0 <= x_int < image_width and 0 <= y_int < image_height:
            heatmap[y_int, x_int] += intensity

    # for coordinate in coordinates:
    #     x = int(coordinate[0])
    #     y = int(coordinate[1])
    #     heatmap[y, x] += intensity

    sigma = 40
    smoothed_heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Placing the frame
    plt.figure(figsize=(10, 10 * image_height / image_width))
    plt.imshow(frame, extent=[0, image_width, 0, image_height], origin='upper')

    # Overlaying the heatmap on top of the frame
    plt.imshow(smoothed_heatmap, cmap='jet', alpha=0.3, extent=[0, image_width, 0, image_height], origin='lower')

    # Plot dimensions
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    
    plt.colorbar(label="Intensity")
    # plt.axis('off')
    plt.show()
