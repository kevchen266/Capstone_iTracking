from queue import Queue
from collections import deque

q = deque(maxlen=2000)
# prediction q ([image, metadata])
prediction_q = deque(maxlen=2000)
# model output queue([video name, frame number, x,y]) goes into db
model_output_q = deque()
# [
#     "video_name" : name,
#     "frame_number" : framenumber,
#     "coordinate" : (x,y)
# ]

# or

# [["video name1", "frame number1", "x1","y1"],["video name2", "frame number2", "x2","y2"]]
heatmap_q = deque()
calibration = []
dataloader_q = deque(maxlen=2000)