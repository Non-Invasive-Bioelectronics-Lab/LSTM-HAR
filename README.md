# LSTM-HAR
An LSTM designed for Human Activity Recognition using Tensorflow 2 on Python 3. LSTM also converted for use on an Android App.

This code is submitted for the dissertation "Design and Optimization of a Deep Learning Neural Network for Human Activity Recognition on a Smartphone" at the University of Manchester. Abstract is detailed below.

### ABSTRACT
Recent successes of deep learning in object detection, speech recognition, and virtual assistants has increased interest in deploying these models on edge devices such as smartphones and wearables. These devices are now capable of collecting large amounts of data about various aspects of a user’s daily life. Edge computing allows for computation at the source of data collection to improve user privacy and reduce the computational budget of these applications for faster, more reliable inference. In this report, a Stacked LSTM structure is designed to extract temporal features from inertial sensor data for Human activity recognition (HAR). This model is trained and optimized offline to achieve an overall accuracy of 92.84% on 6 activities using a public dataset. The effects of adding normalization and regularization techniques are investigated to improve generalization, reduce model overfitting, and accelerate training times. The compatibility of these models for inference on a smartphone device is also explored. The offline model is converted for use on a smartphone and optimized for memory efficient, low-power inference. Quantization techniques are shown to reduce the precision of the model’s weights to achieve a 2.3x model size reduction for improved latency and power consumption. An Android app was designed to successfully implement this model on a smartphone using the Android Emulator. This model showed great promise for commercial application by achieving an overall accuracy of 92.67% with a model size of 27KB for low-power HAR inference. 

### LSTM Model using TensorFlow
The LSTM was built using TensorFlow 2 on Python 3 and the code is found in the LSTM_UCIHAR Folder. The build_LSTM.py file is used to build and save the network. The LSTM_converter.py file is used to convert the model to a tflite file. The LSTM_interpreter.py file is used to run the tflite file using the Python Interpreter.

### Dataset
The LSTM is trained and tested using the UCI-HAR dataset found at: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
This data should be loaded into the correct path when using the code. Two TFlite models are generated, LSTM_float.tflite (float-32) and LSTM_dynamic.tflite (unit-8). This dataset also contains handcrafted features, but these are not used.

### Instructions for Android App using UCI-HAR test data:

1) Load app into Android Studio
2) Select desired model (float-32 or unit-8) in HARClassifier.java
3) Set up Android Emulator device.
4) Run app from MainActivity.java on Android Emulator Device

The app was also contains commented sections of code that can be used to collect data directly from the smartphone's embedded sensors. However, this could not be tested accurately using the Emulator, so the code may need some extra work. 

Sources used to help develop code for loading model on Android Studio:
1) Saeed.A (https://github.com/aqibsaeed/Human-Activity-Recognition-using-CNN)
