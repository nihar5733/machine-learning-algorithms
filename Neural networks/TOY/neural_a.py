import sys
import csv
import math
import numpy as np
from numpy import genfromtxt

# python neural a.py trainﬁle.csv param.txt weightﬁle.txt 

def getXandY(dataset):
	y=dataset[:,-1]
	y=y.reshape([y.shape[0],1])
	x=dataset[:,:-1]
	return x,y

def sigmoid(x):
	return 1/(1+np.exp(-x))

def forward_propagation(x,weights):
	allz=[None]*len(weights)

	z1 = sigmoid(np.dot(x,weights[0]))
	z1 = np.c_[np.ones([z1.shape[0],1]),z1]
	allz[0]=z1
	for i in range(len(weights)-2):
		z1 = sigmoid(np.dot(allz[i],weights[i+1]))
		z1 = np.c_[np.ones([z1.shape[0],1]),z1]
		allz[i+1]=z1
	allz[-1] = sigmoid(np.dot(allz[-2],weights[-1]))

	return allz

def back_propagation(weights,allz,x,y,batch_size):
	l=len(weights)
	gradients=[None]*l
	l=l-1
	e=allz[l]-y
	gradients[l] = np.dot(allz[l-1].T,e)/batch_size

	B=np.dot(e,weights[l].T)/batch_size
	C=(B[:,1:])*(allz[l-1][:,1:])*((1-(allz[l-1]))[:,1:])
	t=l-1
	while t>0:
		gradients[t]=np.dot(allz[t-1].T,C)
		B=np.dot(C,weights[t].T)
		C=(B[:,1:])*(allz[t-1][:,1:])*((1-allz[t-1])[:,1:])
		t=t-1
	gradients[0]=np.dot(x.T,C)
	
	return gradients

def update_weights(weights,gradients,learning_rate):
	for i in range(len(weights)):
		weights[i] = weights[i]- learning_rate*gradients[i]

	return weights

def print_my_weights(weightﬁle_txt,weights):
	file = open(weightﬁle_txt,'a')

	for i in range(len(weights)):
		np.savetxt(file,weights[i].reshape([(weights[i].shape[0])*(weights[i].shape[1]),1]))
	file.close()

trainﬁle_csv = sys.argv[1]
param_txt = sys.argv[2]
weightﬁle_txt = sys.argv[3]

trainﬁle = np.genfromtxt(trainfile_csv,delimiter=',')
trainﬁle=np.c_[np.ones([trainﬁle.shape[0],1]),trainﬁle]

params_file = open(param_txt,"r")
params_str = params_file.readlines()

strategy = int(params_str[0])
learning_rate = float(params_str[1])
max_itr = int(params_str[2])
batch_size = int(params_str[3])
layers=params_str[4].split(',')
layers=np.array(layers,dtype=int)

params_file.close()
if(strategy==1):
	weights = []
	gradients=[]
	no_of_layers = layers.shape[0]
	noOfBatches = math.floor(trainfile.shape[0]/batch_size)
	itr=0
	weights.append(np.zeros([trainfile.shape[1]-1,layers[0]]))
	for i in range(no_of_layers-1):
		weights.append(np.zeros([layers[i]+1,layers[i+1]]))
	weights.append(np.zeros([layers[-1]+1,1]))

	print(weights[0].shape)
	print(weights[1].shape)

	while itr<max_itr:
		for i in range(noOfBatches):
			train_batch = trainﬁle[i*batch_size:(i+1)*batch_size,:]
			x_batch,y_batch=getXandY(train_batch)

			allz = forward_propagation(x_batch,weights)

			gradients = back_propagation(weights,allz,x_batch,y_batch,batch_size)

			weights = update_weights(weights,gradients,learning_rate)
			
			itr+=1
			
			if(itr>=max_itr): break
	print_my_weights(weightﬁle_txt,weights)
	
elif(strategy==2):
	weights = []
	gradients=[]
	no_of_layers = layers.shape[0]
	noOfBatches = math.floor(trainfile.shape[0]/batch_size)
	itr=0
	weights.append(np.zeros([trainfile.shape[1]-1,layers[0]]))
	for i in range(no_of_layers-1):
		weights.append(np.zeros([layers[i]+1,layers[i+1]]))
	weights.append(np.zeros([layers[-1]+1,1]))

	while itr<max_itr:
		for i in range(noOfBatches):
			train_batch = trainﬁle[i*batch_size:(i+1)*batch_size,:]
			x_batch,y_batch=getXandY(train_batch)

			allz = forward_propagation(x_batch,weights)

			gradients = back_propagation(weights,allz,x_batch,y_batch,batch_size)

			weights = update_weights(weights,gradients,learning_rate/math.sqrt(itr))
			
			itr+=1
			
			if(itr>=max_itr): break
	print_my_weights(weightﬁle_txt,weights)
