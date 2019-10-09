import sys
import csv
import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.optimizers import RMSprop
from keras import regularizers
from keras.optimizers import Adam

trainﬁle_csv = sys.argv[1]
testfile_csv = sys.argv[2]
outputﬁle_txt = sys.argv[3]

trainfile = np.loadtxt(trainﬁle_csv,delimiter=' ')

testfile = np.loadtxt(testfile_csv,delimiter=' ')

def getXandY(data):
  y=data[:,-1]
  yf = keras.utils.to_categorical(y, 10)

  xt=data[:,:-1]
  xf=np.zeros([xt.shape[0],3,32,32])
  for i in range(xt.shape[0]):
    xf[i] = xt[i].reshape(3,32,32)
  
  xf=np.transpose(xf,axes=(0,3,2,1))
  return xf,yf


x_train,y_train=getXandY(trainfile)
x_test,y_test=getXandY(newTestFile)
y_test=np.zeros([y_test.shape[0],10])


mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 
num_classes = 10

weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
 
model.summary()
adam = Adam(lr = 0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
print(x_train.shape, y_train.shape)
model.fit(x=x_train, y=y_train,
          epochs=10,
          batch_size=128)
adam = Adam(lr = 0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train,
          epochs=10,
          batch_size=128)
adam = Adam(lr = 0.00025)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train,
          epochs=10,
          batch_size=128)

adam = Adam(lr = 0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
print(x_train.shape, y_train.shape)
model.fit(x=x_train, y=y_train,
          epochs=5,
          batch_size=256)
adam = Adam(lr = 0.00001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
print(x_train.shape, y_train.shape)
model.fit(x=x_train, y=y_train,
          epochs=5,
          batch_size=256)

score = model.evaluate(x_test, y_test, batch_size=128)

y_pred = model.predict(x_test)

def matrix_to_y(ymat):
  y=np.zeros([ymat.shape[0],1])
  for i in range(ymat.shape[0]):
      y[i]=int(np.argmax(ymat[i,:]))
  return y

y_vec_pred = matrix_to_y(y_pred)
np.savetxt(outputﬁle_txt,y_vec_pred, fmt='%d')
