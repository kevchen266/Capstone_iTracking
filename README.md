## CNN Model Workflow
**Personalization**: Calibration adapts the model to individual users for higher accuracy.

**Real-Time Prediction**: The CNN model predicts gaze coordinates at 30 FPS.

**Data Visualization**: Heatmap generation provides intuitive insights into user focus areas.

![截圖 2024-12-09 下午11 03 54](https://github.com/user-attachments/assets/468c7ecf-4105-4c04-b96c-b0ff8620aeb3)




## Application Workflow
[Watch Applicatino video](https://drive.google.com/file/d/1-0CaW9Lbr7oO4c62jiWdgf5DE8O13tmg/view?usp=sharing)

![image](https://github.com/user-attachments/assets/ec4f6462-f331-46ec-90a2-cf9f8cd965bf)


### Calibration Stage (Mini Training)

**Objective**: To capture and preprocess user-specific images, enabling the model to undergo mini training for improved accuracy and personalization of eye-gaze tracking.


### Prediction Stage (CNN Model Eye-Gaze Prediction)

**Objective**: To analyze processed user images and generate precise eye-gaze coordinate predictions using the calibrated CNN model.



### Heatmap Generation Stage (Visualization of Eye-Gaze Prediction)
**Objective**: To generate a heatmap that visualizes the distribution of predicted eye-gaze coordinates, highlighting the regions where the user is focusing based on the model's predictions.


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

## System Architecture
![截圖 2024-09-23 下午2 48 26](https://github.com/user-attachments/assets/2ba8247f-5e3b-45c8-8a7f-a1295d0feb2a)

**Overview**
The system is designed to track user eye-gaze behavior through a three-stage pipeline: Calibration, Prediction, and Heatmap Generation. These stages work together to process user-specific data, predict eye-gaze coordinates, and visualize the results effectively.
