# Face-Detection-with-Attributes-using-Azure-Computer-Vision-API
Main Goal: Detecting faces in an image along with face attributes like age, gender using Azure computer vision API
Details: 
  Here we develop a program using Azure computer vision API to make a near real-time face detection
  using a webcam in python. Azure computer vision is used for face detection and face attributes
  This program analyses video frames from webcam in near real-time using azure computer vision APIs.
  Azure computer vision contains different APIs, but for this program we are using the face detection API
  and extract face attributes such as age and gender.
  We used OpenCV library to capture frames using webcam
  This program reads the stream of frames from webcam and selects which frames should be
  analysed. To decrease the cost of API usage, we are sending one frame per second to the API.
  The selected frames will be sent to the face APIs to detect face/faces along with face attributes
  like age and gender. 
  You need to enter subscription key and endpoint provided to you by Azure Computer Vison inside program to make it work.
  
  Check Details of API here: https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/
