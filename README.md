# Avatar Pose Tracker

This project utilizes MediaPipe and OpenCV to perform real-time pose tracking and render an avatar based on detected pose landmarks.

## Features
- Real-time pose tracking using MediaPipe Pose.
- Custom avatar rendering on a blank canvas.
- Displays both the original video feed and the avatar side-by-side.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shabeer-10/Avatar_Pose_Tracker.git
2. Navigate to the project directory:
   ```bash
   cd Avatar_Pose_Tracker
3. Install the required packages:
   ```bash
   pip install opencv-python mediapipe numpy

## Usage
Run the script:
```
python Avatar_Pose_Tracker.py
```

![Avatar Pose Tracker](avatar_demo.png)

## Contributing  
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request with your changes.  

## Known Issues  
- Avatar tracking may not work well in low-light conditions.  
- Performance may vary depending on webcam resolution and system hardware.  



