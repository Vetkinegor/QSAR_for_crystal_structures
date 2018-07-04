#import basic library
import numpy as np
import pandas as pd
from random import random
import bottleneck as bn
import heapq
#load y_label
docX = []
docY = []
df_out = pd.read_csv('C:/130224Y.csv',header=None)
filename = df_out[0]
Y_data230 = np.array(df_out[1]) - 1
Y_data7 = np.array(df_out[2]) - 1
Y_data101 = np.array(df_out[3]) – 1
#load X_data
for i in filename :
 filepath = "D://JI//newxrd//fresh_random//" + str(i) + ".csv"
 arr = pd.read_csv(filepath,header=None)
 arr = np.array(arr)
docX.append(arr)
X_data = np.array(docX)
docX = []
#randomly choose 20% test data
tot_ix =range(len(Y_data7))
test_ix = np.random.choice(tot_ix, int(len(Y_data7)*0.2), replace=False)
test_ix = np.sort(test_ix,axis=0)
train_ix = list(set(tot_ix) - set(test_ix))
#write test data index into csv files
test_ix = np.reshape(test_ix, test_ix.shape + (1,))
mat1 = test_ix
dataframe1 = pd.DataFrame(data=mat1.astype(int))
dataframe1.to_csv('choose20percenttestset.csv', sep=',', header=False,
float_format = '%.2f', index = False)
#load test index and convert to hot vector
test_index = pd.read_csv('choose20percenttestset.csv', header=None)
test_ix = test_index[0]
tot_ix = range(len(Y_data7))
train_ix = list(set(tot_ix) - set(test_ix))
test_X = X_data[test_ix]
train_X = X_data[train_ix]
test_Y7 = Y_data7[test_ix]
train_Y7 = Y_data7[train_ix]
test_Y101 = Y_data101[test_ix]
train_Y101 = Y_data101[train_ix]
test_Y230 = Y_data230[test_ix]
train_Y230 = Y_data230[train_ix]
from keras.utils.np_utils import to_categorical
train_Y7 = to_categorical(train_Y7, 7)
test_Y7 = to_categorical(test_Y7, 7)
train_Y101 = to_categorical(train_Y101, 101)
test_Y101 = to_categorical(test_Y101, 101)
train_Y230 = to_categorical(train_Y230, 230)
test_Y230 = to_categorical(test_Y230, 230)
#shuffle data before training
tot_ix =range(len(train_X))
rand_ix = np.random.choice(tot_ix, len(train_X), replace=False)
train_X = train_X[rand_ix]
train_Y101 = train_Y101[rand_ix]
train_Y7 = train_Y7[rand_ix]
train_Y230 = train_Y230[rand_ix]
#import keras library
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Merge, merge
from keras.layers import Dropout, Activation
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers import ZeroPadding1D
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
import keras.callbacks
from keras.models import Model
# 7 label training
model = Sequential()
model.add(Convolution1D(80, 100, subsample_length = 5, border_mode =
'same', input_shape=(10001,1))) #add convolution layer
model.add(Activation('relu')) #activation
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=2)) #pooling layer
model.add(Convolution1D(80, 50, subsample_length = 5, border_mode =
'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=None))
model.add(Convolution1D(80, 25, subsample_length = 2, border_mode =
'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=None))
model.add(Flatten())
model.add(Dense(700))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(70))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
# check the accuracy
a = model.evaluate(train_X, train_Y7)
print(a)
a = model.evaluate(test_X, test_Y7)
print(a)
# 101 label training
model = Sequential()
model.add(Convolution1D(80, 100, subsample_length = 5, border_mode =
'same', input_shape=(10001,1))) #add convolution layer
model.add(Activation('relu')) #activation
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=2)) #pooling layer
model.add(Convolution1D(80, 50, subsample_length = 5, border_mode =
'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=None))
model.add(Convolution1D(80, 25, subsample_length = 2, border_mode =
'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=None))
model.add(Flatten())
model.add(Dense(4040))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(202))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(101))
model.add(Activation('softmax'))
#Compile
model.compile(loss='categorical_crossentropy', optimizer='Adam',
metrics=['accuracy'])
#fit
filepath='C:/101labelmodel.out'
modelCheckpoint=keras.callbacks.ModelCheckpoint(filepath,
monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
history = keras.callbacks.History()
model.fit(train_X, train_Y101, batch_size=800, nb_epoch=5000,
validation_split=0.25, callbacks=[modelCheckpoint,history], shuffle=True)
# check the accuracy
a = model.evaluate(train_X, train_Y101)
print(a)
a = model.evaluate(test_X, test_Y101)
print(a)
# 230 label training
model = Sequential()
model.add(Convolution1D(80, 100, subsample_length = 5, border_mode =
'same', input_shape=(10001,1))) #add convolution layer
model.add(Activation('relu')) #activation
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=2)) #pooling layer
model.add(Convolution1D(80, 50, subsample_length = 5, border_mode =
'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=None))
model.add(Convolution1D(80, 25, subsample_length = 2, border_mode =
'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(AveragePooling1D(pool_length=3, stride=None))
model.add(Flatten())
model.add(Dense(2300))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1150))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(230))
model.add(Activation('softmax'))
#Compile
model.compile(loss='categorical_crossentropy', optimizer='Adam',
metrics=['accuracy'])
#fit
filepath='C:/230labelmodel.out'
modelCheckpoint=keras.callbacks.ModelCheckpoint(filepath,
monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
history = keras.callbacks.History()
model.fit(train_X, train_Y230, batch_size=1000, nb_epoch=5000,
validation_split=0.25, callbacks=[modelCheckpoint,history], shuffle=True)
# check the accuracy
a = model.evaluate(train_X, train_Y230)
print(a)
a = model.evaluate(test_X, test_Y230)
print(a)
#save log after training
acc_log = history.history['acc']
val_acc_log = history.history['val_acc']
loss_log = history.history['loss']
val_loss_log = history.history['val_loss']
acc_log = np.array(acc_log)
val_acc_log = np.array(val_acc_log)
loss_log = np.array(loss_log)
val_loss_log = np.array(val_loss_log)
mat = np.vstack((loss_log, acc_log, val_loss_log, val_acc_log))
mat = np.transpose(mat)
dataframe1 = pd.DataFrame(data=mat)
dataframe1.to_csv('save_log.csv', sep=',', header=False,
float_format='%.7f', index=False)
