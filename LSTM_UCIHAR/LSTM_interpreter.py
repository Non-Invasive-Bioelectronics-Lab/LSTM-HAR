import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

#Load Training Data, Total Acc Train

xtrain_acc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt',
                 header = None, delim_whitespace=True))

ytrain_acc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt',
                 header = None, delim_whitespace=True))

ztrain_acc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt',
                 header = None, delim_whitespace=True))

#Body Acc Train
xtrain_bacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt',
                 header = None, delim_whitespace=True))

ytrain_bacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt',
                 header = None, delim_whitespace=True))

ztrain_bacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt',
                 header = None, delim_whitespace=True))

#Gyro Train
xtrain_gyro = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt',
                 header = None, delim_whitespace=True))

ytrain_gyro = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt',
                 header = None, delim_whitespace=True))

ztrain_gyro = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt',
                 header = None, delim_whitespace=True))

#Training Labels
y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt',
                 header = None, delim_whitespace=True)
print(y_train.shape)

#Load test data, Total Acc Test

xtest_acc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt',
                 header = None, delim_whitespace=True))

ytest_acc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt',
                 header = None, delim_whitespace=True))

ztest_acc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt',
                 header = None, delim_whitespace=True))

#Body Acc Test
xtest_bacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt',
                 header = None, delim_whitespace=True))

ytest_bacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt',
                 header = None, delim_whitespace=True))

ztest_bacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt',
                 header = None, delim_whitespace=True))

#Gyro Test
xtest_gyro = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt',
                 header = None, delim_whitespace=True))

ytest_gyro = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt',
                 header = None, delim_whitespace=True))

ztest_gyro = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt',
                 header = None, delim_whitespace=True))

#Test Labels
y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt',
                 header = None, delim_whitespace=True)
print(y_test.shape)

#MinMax Scaling Normalization
def normalize(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train.reshape(-1,1))
    trainN = scaler.transform(train.reshape(-1,1)).reshape(train.shape)
    testN = scaler.transform(test.reshape(-1,1)).reshape(test.shape)
    return trainN, testN

#Scale x
xtrain_accN, xtest_accN = normalize(xtrain_acc, xtest_acc)
xtrain_baccN, xtest_baccN = normalize(xtrain_bacc, xtest_bacc)
xtrain_gyroN, xtest_gyroN = normalize(xtrain_gyro, xtest_gyro)

#Scale y
ytrain_accN, ytest_accN = normalize(ytrain_acc, ytest_acc)
ytrain_baccN, ytest_baccN = normalize(ytrain_bacc, ytest_bacc)
ytrain_gyroN, ytest_gyroN = normalize(ytrain_gyro, ytest_gyro)

#Scale z
ztrain_accN, ztest_accN = normalize(ztrain_acc, ztest_acc)
ztrain_baccN, ztest_baccN = normalize(ztrain_bacc, ztest_bacc)
ztrain_gyroN, ztest_gyroN = normalize(ztrain_gyro, ztest_gyro)

#Test Set
x_test = [xtest_accN, ytest_accN, ztest_accN,
          xtest_baccN, ytest_baccN, ztest_baccN,
          xtest_gyroN, ytest_gyroN, ztest_gyroN]
x_test = np.array(np.dstack(x_test),dtype = np.float32)
print(x_test.shape)

#Zero-Index Test Labels
y_test = y_test-1

#TFlite Interpreter for Inference
interpreter = tf.lite.Interpreter(model_path="LSTM_dynamic.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Run Interpreter in for loop for every test data entry 
accurate_count = 0

for i in range(x_test.shape[0]):
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1,:,:])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predict_label = np.argmax(output_data) #predicted label from interpreter
    
    #check if prediction is correct
    accurate_count += (predict_label == y_test.iloc[i][0]) 

#Overall accuracy for entire test 
accuracy = accurate_count * 1.0 / y_test.size 
print('TensorFlow Lite model accuracy = %.4f' % accuracy)