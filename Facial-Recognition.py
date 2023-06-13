from tensorflow.keras.models import load_model

# Returns a compiled model identical to the previous one
model = load_model('new_model.h5')


import cv2
import numpy as np

# function to get emotion from model prediction
def decode_emotion(prediction):
  # Assuming prediction is a class vector, find the index of the maximum class score
  pred_index = np.argmax(prediction)
  # Use the index to get the class label
  emotion = label_map[pred_index]
  return emotion

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_AREA)  # Resize image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert color image to grayscale
    
    # Normalization (As per your model training preprocessing steps)
    gray = gray / 255.0
    
    # Reshape the image to (1, 256, 256, 1) because model expects 4D tensors (batch_size, height, width, channels)
    gray = np.reshape(gray, (1, 256, 256, 1))
    
    prediction = model.predict(gray)  # model is the loaded model
    label_map = {0: 'happy', 1: 'sad', 2: 'neutral', 3:'surprised'}

    answer = decode_emotion(prediction)
    # Convert your prediction to readable output
    #emotion = decode_emotion(prediction)
    
    # Display the output
    cv2.putText(frame, answer, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
