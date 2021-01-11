# Wrong-direction-drivers-detection
This project is to detect cars that move in wrong way. Tensorflow implementation based on YOLOv3 detector and Kalman Filtering for the paper: [Real-Time, Deep Learning Based Wrong Direction Detection](https://www.mdpi.com/2076-3417/10/7/2453)
# Abstract
In this paper, we developed a real-time intelligent transportation system (ITS) to detect vehicles traveling the wrong way on the road. The concept of this wrong-way system is to detect such vehicles as soon as they enter an area covered by a single closed-circuit television (CCTV) camera. After detection, the program alerts the monitoring center and triggers a warning signal to the drivers. The developed system is based on video imaging and covers three aspects: detection, tracking, and validation. To locate a car in a video frame, we use a deep learning method known as you only look once version 3 (YOLOv3). Therefore, we used a custom dataset for training to create a deep learning model. After estimating a car’s position, we implement linear quadratic estimation (also known as Kalman filtering) to track the detected vehicle during a certain period. Lastly, we apply an “entry-exit” algorithm to identify the car’s trajectory, achieving 91.98% accuracy in wrong-way driver detection.
# Features
1. Real-time detection (up to 35 FPS on Nvidia 1080Ti)
2. Support parallel training
3. Video and data is included with annotated files.

# Getting started
## Requirenments installation
This project is done on Windows 10 machine with already installed Anaconda platform. If you don't have, you can install from [here](https://www.anaconda.com/products/individual)
Creating an enviorenment can be done by running.
```shell script
conda env create -f requirenments.yml
```
### Data
To run YOLOv3 model you need to download model data
- [x] Download pre-trained model
