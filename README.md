## System Architecture
![截圖 2024-09-23 下午2 48 26](https://github.com/user-attachments/assets/2ba8247f-5e3b-45c8-8a7f-a1295d0feb2a)


### Calibration Stage

**Objective**: To send user's calibration images and data to the machine learning model for training purposes.

1. **User Setup**: The user positions their head in a frame and looks at different spots for calibration.
2. **Data Collection**: The frontend sends real-time images to the backend at 30 frames per second.
3. **Image Processing**: The backend processes these images, crops the eye area, and sends them to the machine learning model for training. The goal is to learn the user's eye movement patterns.

### Prediction Stage

**Objective**: To predict the eye gaze spot coordinates based on the trained model.

1. **Data Collection**: Similar to the calibration stage, the frontend sends real-time images to the backend at 30 frames per second.
2. **Image Processing**: The backend processes the images, crops the eye area, and sends them to the machine learning model for prediction. The model predicts the eye gaze spot coordinates and display heatmap on video.
3. **Video Playback**:  The backend sends the corresponding video URL to the frontend for playback.

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
