## CNN Model Architecture
This CNN-based architecture predicts gaze coordinates (x, y) using facial and eye images from the [MPIIGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild)
. Left and right eye images are processed independently through identical CNNs with convolutional layers to extract spatial features, pooling layers to reduce dimensions, and fully connected layers (FC-F1 and FC-FC1) to create dense representations. The face image follows a similar pipeline, capturing global context like head orientation. These features are fused into a single vector, passed through fully connected layers, and output gaze predictions. The MPIIGaze dataset provides eye and face images with gaze annotations, enabling the model to map visual features to gaze coordinates effectively.

![截圖 2024-12-09 下午11 03 54](https://github.com/user-attachments/assets/468c7ecf-4105-4c04-b96c-b0ff8620aeb3)




## **Application Demo**
[Watch Applicatino video](https://drive.google.com/file/d/1-0CaW9Lbr7oO4c62jiWdgf5DE8O13tmg/view?usp=sharing)




### **Calibration Stage (Mini Training)**

**Objective**: To capture and preprocess user-specific images, enabling the model to undergo mini training for improved accuracy and personalization of eye-gaze tracking.

**Instructions**:
After Calibration, the console will prompt you to specify the movie you want to play. For example:
```sh
Enter the movie URL: http://172.31.231.33:8000/videos/001_h264_1K.mp4

```
Make sure to provide the correct movie URL when prompted. This video URL corresponds to the video name defined in the `handle_video_request(self, num_videos)` function in the `consumer` file.

### Prediction Stage (CNN Model Eye-Gaze Prediction)

**Objective**: Analyze processed user images to predict precise eye-gaze coordinates using a calibrated CNN model. During this stage, the user watches a video while eye images are captured, processed, and input into the CNN model for real-time gaze prediction.


### Heatmap Generation Stage (Visualization of Eye-Gaze Prediction)
**Objective**: To generate a heatmap that visualizes the distribution of predicted eye-gaze coordinates, highlighting the regions where the user is focusing based on the model's predictions.

![image](https://github.com/user-attachments/assets/ec4f6462-f331-46ec-90a2-cf9f8cd965bf)


## **Thread Management**

The application utilizes multithreading to manage different stages concurrently within the WebSocket `consumer`. Threads are initiated in the `connect` method of the WebSocket consumer to ensure tasks run in parallel:

1. **Calibration Thread**:  
   Starts multiple workers for prediction tasks:
   ```python
   start_calibration_workers()

1. **Prediction Thread**:  
   Starts multiple workers for prediction tasks:
   ```python
   start_prediction_workers(num_workers=4)
2. **Heatmap Generation Thread**
    Starts multiple workers for prediction tasks:
   ```python
   start_heatmap_worker()
## Setting Up the Environment

1. Clone the repository:

```sh
    git clone <repository_url>
    cd eye_tracking_project
```

2. Create and activate a virtual environment:

```sh
    python3 -m venv new_env
    source new_env/bin/activate
```

3. Install dependencies:

```sh
    pip install -r requirements.txt
```

## Running the Server

1. Configure the frontend code in Android Studio with the correct IP and URL.

2. Start the backend Django server using Daphne:

    ```sh
    daphne -p 8000 -b 0.0.0.0 eye_tracking_project.asgi:application
    ```

3. Ensure the frontend and backend are correctly communicating over the configured IP and URL.

## Data Workflow
![截圖 2024-09-23 下午2 48 26](https://github.com/user-attachments/assets/2ba8247f-5e3b-45c8-8a7f-a1295d0feb2a)

**Overview**
The system is designed to track user eye-gaze behavior through a three-stage pipeline: Calibration, Prediction, and Heatmap Generation. These stages work together to process user-specific data, predict eye-gaze coordinates, and visualize the results effectively.
