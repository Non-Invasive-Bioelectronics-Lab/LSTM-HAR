import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

#Load Pre-Trained Model
model = tf.keras.models.load_model('saved_models/LSTM_Final')

#Check model architecure
model.summary()

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

#One-hot encode test labels
y_test = y_test-1
y_test = tf.keras.utils.to_categorical(y_test)
print(y_test.shape)

#Test Model
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size = 64)
print('Restored model, accuracy: {:5.2f}%'.format(100*test_acc))

#Convert to TF Lite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/LSTM_Final')
converter.experimental_new_quantizer = True
#converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uncomment to optimize model weights

#Uncomment below for full integer quantization (Not supported for LSTMs currently)
''' 
def representative_data_gen(): 
    for input_data in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
        yield [input_data]
  
converter.representative_dataset = representative_data_gen 
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
'''

#Convert and Save Model
tflite_model = converter.convert()

with tf.io.gfile.GFile('LSTM_dynamic2.tflite','wb') as f:
    f.write(tflite_model)
