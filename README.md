# AccidentDetection
This project demonstrates a video-based detection system that tracks a red car moving from the right side of the screen and a pedestrian walking.

## Features
- Tracks the red car in the video.
- Tracks the pedestrian.
- Displays the distance between the red car and the pedestrian.
- Alerts when a potential accident is detected.

## Method
1. Track the car and pedestrian using YOLOv11 pretrained with the COCO dataset.
2. Filter only the red car by calculating the mean color inside the bounding box.
3. Use the last known position if the object cannot be found in the current frame.
4. Draw a line between the center of the red car and the pedestrian and calculate the Euclidean distance.
5. Alert with a red border when a potential accident is detected.

## Dependencies
Ensure you have Python installed, then install the required dependencies:
```
pip install ultralytics
pip install opencv-python numpy
```

## Usage 
Run the detection script on your video input.
```
python main.py
```