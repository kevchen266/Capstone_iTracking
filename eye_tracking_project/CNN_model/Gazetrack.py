import torch
import torch.nn as nn

class Gazetrack(nn.Module):
    """
    Model Structure:
    1. **Left Eye Convolutional Layers (self.left_eye_conv)**:
        - Input: RGB image with shape (batch_size, 3, 224, 224)
        - Feature extraction: 5 convolutional layers with ReLU activation and max pooling.
        - Output: Feature maps with shape (batch_size, 512, 28, 28)

    2. **Right Eye Convolutional Layers (self.right_eye_conv)**:
        - Same structure and parameters as the left eye convolutional layers.

    3. **Fully Connected Layers for Eye Features (self.fc_eye)**:
        - Flattened feature maps are reduced to a 128-dimensional feature vector.
        - Includes ReLU activation and a 20% Dropout layer to prevent overfitting.

    4. **Final Fully Connected Layers (self.fc)**:
        - Concatenates the left and right eye feature vectors (256-dimensional combined vector).
        - Performs regression to output 2D gaze coordinates (x, y).
    """
    def __init__(self):
        super(Gazetrack, self).__init__()
        def create_eye_conv():
            return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),    # kernel size 5x5
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                   # Output: 64 x 112 x 112

                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # kernel size 5x5
                nn.ReLU(),

                nn.MaxPool2d(kernel_size=2, stride=2),                   # Output: 128 x 56 x 56

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # kernel size 3x3
                nn.ReLU(),


                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # kernel size 3x3
                nn.ReLU(),


                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # kernel size 3x3
                nn.ReLU()
            )

        self.left_eye_conv = create_eye_conv()
        self.right_eye_conv = create_eye_conv()

        # Fully connected layer for eye features with ReLU
        self.fc_eye = nn.Sequential(
            nn.Linear(512 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # last fc layer, concatenate left and right eye.
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),  # Dropout layer with 30% dropout rate
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.2),  # Dropout layer with 30% dropout rate
            nn.Linear(32, 2)
        )

    def forward(self, left_eye, right_eye):
        # Pass both eyes through the shared convolutional layers
        left_eye_features = self.left_eye_conv(left_eye)
        right_eye_features = self.right_eye_conv(right_eye)

        # Print the feature map shapes for debugging
        # print(f"Left Eye Conv Output Shape: {left_eye_features.shape}")
        # print(f"Right Eye Conv Output Shape: {right_eye_features.shape}")

        # Flatten the features
        left_eye_features = left_eye_features.view(left_eye_features.size(0), -1)
        right_eye_features = right_eye_features.view(right_eye_features.size(0), -1)


        # print(f"Left Eye Flattened Shape: {left_eye_features.shape}")
        # print(f"Right Eye Flattened Shape: {right_eye_features.shape}")

        # Pass through fully connected layers for each eye
        left_eye_features = self.fc_eye(left_eye_features)
        right_eye_features = self.fc_eye(right_eye_features)


        # print(f"Left Eye FC Output Shape: {left_eye_features.shape}")
        # print(f"Right Eye FC Output Shape: {right_eye_features.shape}")

        # Concatenate features and pass through final layers
        combined_features = torch.cat((left_eye_features, right_eye_features), dim=1)
        output = self.fc(combined_features)


        # print(f"Output Shape: {output.shape}")

        return output
