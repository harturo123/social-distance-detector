# Using AI to measure social distancing in CCTV video

## Goal
This is an AI model that measures the distance between any pair of people appearing on CCTV video and detects cases where the Covid-19 social distance is not maintained.

The model can be easily callibrated to use video from any height and angle.

![image](https://user-images.githubusercontent.com/45198217/118699202-8e857d80-b811-11eb-91fb-df7d40e12df7.png)

## Implementation
Raw video from the CCTV camera is transformed into a bird's-eye view using an homography transformation implemented in OpenCV, a Python library for computer vision. The transformation is callibrated selecting 4 points in the image that correspond to a rectangle in the real world.

![image](https://user-images.githubusercontent.com/45198217/118703135-0a81c480-b816-11eb-88fb-de2bf9cbe3ae.png) ![image](https://user-images.githubusercontent.com/45198217/118703436-5fbdd600-b816-11eb-96e5-b197e5f601a3.png)

Then, people is detected using OpenCV HOG detector, and bounding boxes drawn around them. Finally, the distance between any pair of people is computed and, if smaller than the recommended social distancing, the bounding box is colored in red.



## Results
Check video in Youtube: https://youtu.be/gLbaBOfU3SY
