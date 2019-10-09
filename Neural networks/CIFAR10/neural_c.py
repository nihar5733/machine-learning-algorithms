import sys
import csv
import math
import numpy as np
from numpy import genfromtxt
# from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# python neural a.py trainﬁle.csv param.txt weightﬁle.txt 

def getXandY(dataset):
	y=dataset[:,-1]
	y=y.reshape([y.shape[0],1])
	x=dataset[:,:-1]
	return x,y

def sigmoid(x):
	return 1/(1+np.exp(-x))

def dsigmoid(x):
	return x*(1-x)

def softmax(a):
	mat = np.exp(a)
	mat=(mat.T/sum(mat.T)).T

	return mat

def relu(x):
    return x.clip(min=0)

def drelu(x):
    x.clip(min=0)
    x[x > 0] = 1
    return x

def forward_propagation(x,weights,activation="sigmoid"):
	allz=[None]*len(weights)
	act = eval(activation)
	z1 = act(np.dot(x,weights[0]))
	z1 = np.c_[np.ones([z1.shape[0],1]),z1]
	allz[0]=z1
	for i in range(len(weights)-2):
		z1 = act(np.dot(allz[i],weights[i+1]))
		z1 = np.c_[np.ones([z1.shape[0],1]),z1]
		allz[i+1]=z1
	allz[-1] = softmax(np.dot(allz[-2],weights[-1]))

	return allz

def back_propagation(weights,allz,x,y,batch_size,activation="sigmoid"):

	act=eval(activation)
	dact=eval("d"+activation)



	l=len(weights)
	gradients=[None]*l
	l=l-1
	e=allz[l]-y
	gradients[l] = np.dot(allz[l-1].T,e)/batch_size

	B=np.dot(e,weights[l].T)/batch_size
	C=(B[:,1:])*dact(allz[l-1][:,1:])
	t=l-1
	while t>0:
		gradients[t]=np.dot(allz[t-1].T,C)
		B=np.dot(C,weights[t].T)
		C=(B[:,1:])*dact(allz[t-1][:,1:])
		t=t-1
	gradients[0]=np.dot(x.T,C)
	
	return gradients


def update_weights(weights,gradients,learning_rate):
	for i in range(len(weights)):
		weights[i] = weights[i]- learning_rate*gradients[i]

	return weights

def y_to_matrix(y):
	ymat = np.zeros([y.shape[0],10])
	for i in range(y.shape[0]):
		ymat[i,int(y[i])]=1
	return ymat
	
def matrix_to_y(ymat):
    y=np.zeros([ymat.shape[0],1])
    for i in range(ymat.shape[0]):
        y[i]=np.argmax(ymat[i,:])
    return y


def print_my_weights(weightﬁle_txt,weights):
	file = open(weightﬁle_txt,'a')

	for i in range(len(weights)):
		np.savetxt(file,weights[i].reshape([(weights[i].shape[0])*(weights[i].shape[1]),1]))
	file.close()

trainﬁle_csv = sys.argv[1]
testfile_csv = sys.argv[2]
outputﬁle_txt = sys.argv[3]

trainﬁle = np.loadtxt(trainfile_csv,delimiter=',')
trainﬁle=np.c_[np.ones([trainﬁle.shape[0],1]),trainﬁle]
x,y=getXandY(trainfile)
y=y_to_matrix(y)

# x=preprocessing.scale(x)

scaler = StandardScaler()
# scaler = MinMaxScaler()

x = scaler.fit_transform(x)

strategy = 2
learning_rate = 0.1
max_itr = 10000
batch_size = 128
# layers=np.array([math.floor((x.shape[1]+10)/2)])
layers=np.array([250])


weights = []
gradients=[]
no_of_layers = layers.shape[0]
noOfBatches = math.floor(x.shape[0]/batch_size)
itr=0

# 	weights.append(np.zeros([trainfile.shape[1]-1,layers[0]]))
# 	for i in range(no_of_layers-1):
# 		weights.append(np.zeros([layers[i]+1,layers[i+1]]))
# 	weights.append(np.zeros([layers[-1]+1,10]))

weights.append(np.random.randn(x.shape[1],layers[0])*np.sqrt(2/x.shape[1]))
for i in range(no_of_layers-1):
	weights.append(np.random.rand(layers[i]+1,layers[i+1])*np.sqrt(2/(layers[i]+1)))
weights.append(np.random.rand(layers[-1]+1,10)*np.sqrt(2/(layers[-1]+1)))

while itr<max_itr:
	for i in range(noOfBatches):
		itr+=1

		x_batch = x[i*batch_size:(i+1)*batch_size,:]
		y_batch=y[i*batch_size:(i+1)*batch_size,:]

		allz = forward_propagation(x_batch,weights,activation="relu") 			
		gradients = back_propagation(weights,allz,x_batch,y_batch,batch_size,activation="relu")

		if (strategy==1):
			weights = update_weights(weights,gradients,learning_rate)
		elif (strategy==2):
			weights = update_weights(weights,gradients,learning_rate/math.sqrt(itr))
		
		if(itr>=max_itr): break
			
			
testfile = np.loadtxt(testfile_csv,delimiter=',')
testfile=np.c_[np.ones([testfile.shape[0],1]),testfile]
testfile=testfile[:,:-1]

# testfile = preprocessing.scale(testfile)

testfile = scaler.fit_transform(testfile)

# y_estimated = forward_propagation_sigmoid(testfile,weights)
y_estimated = forward_propagation(testfile,weights,activation="relu")

y_estimated=y_estimated[-1]

y_vector = matrix_to_y(y_estimated)


np.savetxt(outputfile_txt, y_vector)

