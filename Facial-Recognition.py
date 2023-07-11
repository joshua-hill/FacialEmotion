from tensorflow.keras.models import load_model
model = load_model('new_model.h5')
import cv2
import numpy as np



# function to get emotion from model prediction
def decode_emotion(prediction):
  pred_index = np.argmax(prediction)
  emotion = label_map[pred_index]
  return emotion





cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (48, 48), interpolation = cv2.INTER_AREA)  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert color image to grayscale
    
    # Normalization
    gray = gray / 255.0
    gray = np.reshape(gray, (1, 48, 48, 1))
    
    prediction = model.predict(gray)  # model is the loaded model
    label_map = {0: 'happy', 1: 'sad', 2: 'neutral', 3:'surprised'}

    answer = decode_emotion(prediction)
    
    # Display the output
    cv2.putText(frame, answer, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
