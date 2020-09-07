import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

#Load Training Datasets

#Total acc Train
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

#Train and test set
x_train = [xtrain_accN, ytrain_accN, ztrain_accN,
           xtrain_baccN, ytrain_baccN, ztrain_baccN,
           xtrain_gyroN, ytrain_gyroN, ztrain_gyroN]
x_train = np.array(np.dstack(x_train),dtype=np.float32)
print(x_train.shape)

x_test = [xtest_accN, ytest_accN, ztest_accN,
          xtest_baccN, ytest_baccN, ztest_baccN,
          xtest_gyroN, ytest_gyroN, ztest_gyroN]
x_test = np.array(np.dstack(x_test),dtype = np.float32)
print(x_test.shape)

#Zero indexing for labels
y_train = y_train-1
y_test = y_test-1


#One-hot label encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#Build LSTM Model
time_steps = x_train.shape[1]
features = x_train.shape[2]
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, input_shape = (time_steps,features),return_sequences=True))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(6, activation ='softmax'))

#Model compiler settings
model.compile(optimizer = tf.keras.optimizers.Adam(0.003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Train Model
history = model.fit(x_train,y_train, epochs = 50, batch_size = 64,
          validation_split = 0.2, shuffle = True)

#Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size = 64)
model.summary()

#Save Model

run_model = tf.function(lambda x: model(x))
BATCH_SIZE = 1
STEPS = 128
INPUT_SIZE = 9
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))
model.save('saved_models/LSTM_model',signatures = concrete_func)

#Plot Accuracy and Loss Performance
def plot_graphs(history,metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric],'')
    plt.title('Training and Validation '+metric.capitalize()) #uppercase metric?
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric,'val_'+metric])
    plt.show()
    
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


#Plot Confusion Matrix of Test Predictions
predictions = model.predict(x_test)
Activities = ['Walking', 'Walking Upstairs', 'Walking Downstairs',
              'Sitting', 'Standing', 'Laying']

def plot_CM(true_labels, predictions, activities): 
    max_true = np.argmax(true_labels, axis = 1)
    max_prediction = np.argmax(predictions, axis = 1)
    CM = confusion_matrix(max_true, max_prediction)
    plt.figure(figsize=(16,14))
    sns.heatmap(CM, xticklabels = activities, yticklabels = activities,
                annot = True, fmt = 'd',cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_CM(y_test,predictions, Activities)
print('\nTest accuracy: ', test_acc)
