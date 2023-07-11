# FacialEmotion

In The .ipynb file, multiple various CNNs are trained and evaluated. There are 3 CNN architectures, and 3 transfer learning models (VGG16, ResNet v2, and EfficientNet). The model that had the highest accuracy is then downloaded. This model is then uploaded to the 'facial-recognition.py' file. 

WIP:
This program uses OpenCV to capture an image from your webcam. This image is then processed into an array, and is used to make a prediction on the facial emotion using the uploaded CNN model.

Future Changes:
  Model Training:
  - Test new architectures
  
  Model Application:
  - Track the face using openCV so a higher quality image can be put into the model
  - Add real-time emotion recognition
