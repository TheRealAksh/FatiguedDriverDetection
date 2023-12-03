# DrowsyDriverDetection
This project involves the application of principles from Computer Vision and Deep Learning to identify driver's fatigue/drowsiness, triggering an alert when such a state is detected.

• Constructed a drowsiness detection model for drivers utilizing real-time Eye-Tracking in videos through Haar Cascades and the CamShift algorithm.

• Employed significant features extracted by a Convolutional Neural Network (CNN) from the final pooling layer for each video frame, stitching them into a sequence of feature vectors for consecutive frames.

• Utilized this sequence (2048-D) as input for Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN), predicting driver drowsiness based on the video sequence and activating an alarm when necessary.

• Enhanced network weights through the Adam Optimization algorithm.

Technologies employed: Python 2.7, OpenCV 3.3.0, Tensorflow, Keras, CNN, RNN, LSTM.

Procedure for executing this project:

1) Execute the run_extract_eyes.sh program to monitor eyes in various videos (training data) and store eye patches in a designated folder for each video (Alert and Drowsy).
2) Utilize the training data to retrain the CNN model (Inception V3 model).
3) Execute extract_features.py to derive features from the second last layer of the CNN model, producing a 2048-dimensional vector, and creating a frame sequence as a singular vector for input into the LSTM, a component of Recurrent Neural Networks (RNN).
4) Execute data.py and models.py.
5) Finally, run train.py to obtain predictions for the test data sequence, triggering an alarm if the model identifies the sequence as indicative of drowsiness.
