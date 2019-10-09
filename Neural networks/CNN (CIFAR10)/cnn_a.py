import sys
import csv
import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.optimizers import RMSprop

trainﬁle_csv = sys.argv[1]
testfile_csv = sys.argv[2]
outputﬁle_txt = sys.argv[3]

trainfile = np.loadtxt(trainﬁle_csv,delimiter=' ')

testfile = np.loadtxt(testfile_csv,delimiter=' ')


def getXandY(data):
  y=data[:,-1]
  yf = keras.utils.to_categorical(y, 10)

  xt=data[:,:-1]
  xf=np.zeros([xt.shape[0],32,32,3])
  for i in range(xt.shape[0]):
    xf[i] = xt[i].reshape(32,32,3)
  
  return xf,yf

x_train,y_train=getXandY(trainfile)
x_test,y_test=getXandY(testfile)
y_test=np.zeros([y_test.shape[0],10])


model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(32,32,3),strides=(1, 1), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid"))
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid"))
model.add(Flatten())
model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))


rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])
print(x_train.shape, y_train.shape)
model.fit(x=x_train, y=y_train,
          epochs=20,
          batch_size=128)
 
score = model.evaluate(x_test, y_test, batch_size=128)

y_pred = model.predict(x_test)

def matrix_to_y(ymat):
  y=np.zeros([ymat.shape[0],1])
  for i in range(ymat.shape[0]):
      y[i]=int(np.argmax(ymat[i,:]))
  return y

y_vec_pred = matrix_to_y(y_pred)
np.savetxt(outputﬁle_txt,y_vec_pred, fmt='%d')