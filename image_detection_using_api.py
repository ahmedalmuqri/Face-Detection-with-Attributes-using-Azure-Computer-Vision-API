
"""Here we develop a program using Azure computer vision API to make a near real-time face detection
  using a webcam in python. Azure computer vision is used for face detection and face attributes
  This program analyses video frames from webcam in near real-time using azure computer vision APIs.
  Azure computer vision contains different APIs, but for this program we are using the face detection API
  and extract face attributes such as age and gender.
  We used OpenCV library to capture frames using webcam
  This program reads the stream of frames from webcam and selects which frames should be
  analysed. To decrease the cost of API usage, we are sending one frame per second to the API.
  The selected frames will be sent to the face APIs to detect face/faces along with face attributes
  like age and gender
"""
import os
import sys
import requests
# If you are using a Jupyter notebook, uncomment the following line.
# % matplotlibinline
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw
import cv2


# if receiving input through os module, comment these two i.e. if you want to enter
# subscription key and endpoint through terminal, otherwise for use in jupyter
# notebook, these two lines would be useful for entering key and endpoint
# % env COMPUTER_VISION_SUBSCRIPTION_KEY = 21a177ddce0f47be92fa6549dd73be41
# % env COMPUTER_VISION_ENDPOINT = https: // computervisionsagar.cognitiveservices.azure.com /


def getRectangle(faceDictionary, k):
    # print(faceDictionary)     #for debugging
    rect = faceDictionary["faces"][k]['faceRectangle']
    left = rect['left']
    top = rect['top']
    bottom = left + rect['height']
    right = top + rect['width']
    return ((left, top), (bottom, right))


def draw_face(img, analysis_response):
    faces_response = analysis_response
    #    print(faces)  #for debugging

    output_image = Image.open(BytesIO(img))
    # For each face returned use the face rectangle and draw a red box.
    draw = ImageDraw.Draw(output_image)
    for j in range(len(faces_response["faces"])):
        draw.rectangle(getRectangle(faces_response, j), outline='red')
    return output_image


# Comment these two lines before if you want to enter values through terminal
# This is subscription key and endpoint for intrinsic entry through code
subscription_key = "Enter_your_key_here"
endpoint = "https://computervisionsagar.cognitiveservices.azure.com/"

# Add your Computer Vision subscription key and endpoint to your environment variables.
# Uncomment these 7 lines when entering values through terminal
# if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
#    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
# else:
#    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
#    sys.exit()
# if 'COMPUTER_VISION_ENDPOINT' in os.environ:
#    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']

# URL to be used in post request
analyze_url = endpoint + "vision/v2.1/analyze"
# This will become like this one shown below:
# analyze_url="https://computervisionsagar.cognitiveservices.azure.com/vision/v2.1/analyze"

# Set image_path to the local path of an image that you want to analyze.
# If you want to pass image instead of using webcam, uncomment below code lines not the comments:
# image_path = "C:/Documents/ImageToAnalyze.jpg"
# Read the image into a byte array
# image_data = open(image_path, "rb").read()
# plt.imshow(image)
# plt.axis("off")
# _ = plt.title(image_caption, size="x-large", y=-0.1)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if not cap.isOpened():
    print("Error Opening File")
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        # This would cause a delay of 1 sec so as to pass 1 frame per sec to post request
        cv2.waitKey(1000)
        # Code below would encode the frame(numpynd array) to bmp form
        # This bmp form frame would be furthur converted to bytes datatype
        # to be passed into post request as argument
        retval, frame2 = cv2.imencode('.bmp', frame)
        # Read the image into a byte array
        # Instead of .tobytes(), we can also use tostring()
        # image_data = frame2.tostring()
        image_data = frame2.tobytes()

        headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                   'Content-Type': 'application/octet-stream'}
        # use json instead of octet-stream if you want to use some url as image path

        params = {'visualFeatures': 'Faces'}
        # Below line of params contains more features.
        # params = {'visualFeatures': 'Categories,Description,Color,Faces'}
        response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
        response.raise_for_status()

        analysis = response.json()
        # The 'analysis' object contains various fields that describe the image. The most
        # relevant features/properties/attributes can be checked from API documentation .

        # print(analysis)
        if analysis["faces"] == []:
            print("No person present")
        else:
            # Code below would extract age and gender of all person in analysis
            image_age = []
            image_gender = []
            for i in range(len(analysis["faces"])):
                image_age.append(analysis["faces"][i]['age'])
                image_gender.append(analysis["faces"][i]['gender'])
            # Display the image and overlay it with the caption.
            # image = Image.open(BytesIO(image_data))
            # This loop would print ages and genders of all people in frames
            for i in range(len(image_age)):
                print("Person {} age is {}".format((i + 1), image_age[i]))
                print("Person {} gender is {}".format((i + 1), image_gender[i]))

            image = draw_face(image_data, analysis)
            image=image.convert('RGB')
            open_cv_image = np.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            #open_cv_image = np.array(image) 
            #image.show('frame3')
            cv2.imshow('Frame2', open_cv_image)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
