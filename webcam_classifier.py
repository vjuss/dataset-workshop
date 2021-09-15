import cv2
from fastai.vision.all import *

# Setup: Create an empty folder on your laptop. In that folder, create and activate a Python virtual environment by following https://docs.python.org/3/tutorial/venv.html
# In your command line, the virtual environment being activated, run: pip install python-opencv, pip install pytorch, pip install fastai 
# The virtual env is important as we don't want these heavy libraries globally on our computer
# Place this file and an empty folder "model" in the main folder.
# Run the app: python webcam_classifier.py (enter this in terminal)

# This inference is based on https://github.com/jimmiemunyi/Sign-Language-App/blob/main/webcam_inference.py
# mixed with https://github.com/vinaykudari/mask-classification/tree/master/inference 
# and https://forums.fast.ai/t/get-top-5-predictions-in-image-classifier-solved/76249/4




# Loading the model
model_pkl = 'model/yourmodel.pkl' # Edit this filename to suit your own. We have put our model in the "model" folder and access it here
learn = load_learner(model_pkl)

# Capture video on the webcam
cap = cv2.VideoCapture(0)

# Get the dimensions on the frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Get the area that we want the model to analyze. In this case: almost the whole frame. 
def wanted_area(img):
    area = img[50:(frame_width-50), 50:(frame_height-50)] # was 50:324
    area = cv2.resize(area, (224,224))
    return area


# Making predictions

while True:
    # Capture each frame of the video
    _, frame = cap.read()

    # Flip frame to make it feel more natural
    frame = cv2.flip(frame, flipCode = 1)

    # Draw a blue rectangle to show the area that is analyzed
    cv2.rectangle(frame, (50, 50), (frame_width-50, frame_height-50), (255, 0, 0), 2)

    # Get the image with our helper function
    inference_image = wanted_area(frame)

    # Get the current prediction 
    pred_class, pred_idx, confindences = learn.predict(inference_image) # predict returns a tuple like: ('past', tensor(1), tensor([0.0476, 0.9524])). we make those into variables   
    top_conf, i = confindences.topk(1) #highest confidence of the 2 values. topk(2) would give us both
    top_conf_value = top_conf.numpy()[0] #we only want the confidence number value, not text with "tensor"

    prediction_output = f'The class is {pred_class} with confidence {top_conf_value}'

    # Show results as text
    cv2.putText(frame, prediction_output, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Classification', frame)

    # Press `q` to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture() to move on 
cap.release()

# Close all frames and video windows
cv2.destroyAllWindows()