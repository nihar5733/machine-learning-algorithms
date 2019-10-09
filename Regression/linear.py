import sys
import csv
import math
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

def getXandY(dataset):
	y=dataset[:,-1]
	x=dataset[:,:-1]
	x=np.c_[np.ones(x.shape[0]),x]
	return x,y

def moorePenrose(x,y):
	xt=x.T
	w=xt.dot(x)
	w=np.linalg.inv(w)
	w=w.dot(xt)
	w=w.dot(y)
	return w

def ridgeReggresion(x,y,lamda):
	xt=x.T
	w=xt.dot(x)
	w=w+(lamda*np.identity(xt.shape[0]))
	w=np.linalg.inv(w)
	w=w.dot(xt)
	w=w.dot(y)
	return w

def testDataOutput(testfile_csv, outputfile_txt, w):
	x_test=np.array(genfromtxt(testfile_csv,delimiter=','))
	x_test=np.c_[np.ones(x_test.shape[0]), x_test]
	y_test=x_test.dot(w)
	np.savetxt(outputfile_txt,y_test)

def kFoldLasso(train_data, params, k):
	errors = np.zeros(params.shape)
	
	height = math.floor(train_data.shape[0]/k)
	i=0
	for lamda in params:
		error = 0
		counter=0
		while(counter<10):
			k_train_data = np.delete(train_data,slice(counter*height, (counter+1)*height),0)
			k_verif_data = train_data[counter*height:(counter+1)*height,:]
			
			x_train, y_train = getXandY(k_train_data)

			model = linear_model.LassoLars(alpha=lamda)
			model.fit(x_train,y_train)
			
			x_verif, y_verif_true=getXandY(k_verif_data)
			
			y_verif_estimated=model.predict(x_verif)
			error = (error*counter+ np.linalg.norm(y_verif_estimated - y_verif_true)**2/np.linalg.norm(y_verif_true)**2)/(counter+1)
			
			counter=counter+1

		print(error)
		errors[i] = error
		i=i+1

	my_lamda = params[np.argmin(errors)]

	print(my_lamda)
	return my_lamda

def createFeatures(x):
	#x=preprocessing.scale(x)
	poly=PolynomialFeatures(degree=2,interaction_only=False)

	x=poly.fit_transform(x)
	x=preprocessing.scale(x)
	return x

def getNonZeroCols(x,nzc):
    n=len(nzc)
    newx=np.ones(x.shape[0])
    for i in range(n):
        newx=np.c_[newx,x[:,nzc[i]]]
    
    return newx

mode=sys.argv[1]
arguments=sys.argv[2:]

if(mode=='a'):

	trainfile_csv = arguments[0]
	testfile_csv = arguments[1]

	outputfile_txt = arguments[2]
	weightfile_txt = arguments[3]

	train_data=np.array(genfromtxt(trainfile_csv,delimiter=','))

	x_train, y_train = getXandY(train_data)

	w=moorePenrose(x_train,y_train)

	np.savetxt(weightfile_txt,w)

	testDataOutput(testfile_csv,outputfile_txt,w)

elif(mode=='b'):

	k=10

	trainfile_csv = arguments[0]
	testfile_csv = arguments[1]

	regularization_txt = arguments[2]
	outputfile_txt =  arguments[3]
	weightfile_txt =  arguments[4]

	train_data=np.array(genfromtxt(trainfile_csv,delimiter=','))

	params = np.loadtxt(regularization_txt)

	errors = np.zeros(params.shape)

	height = math.floor(train_data.shape[0]/k)
	i=0
	for lamda in params:
		error = 0
		counter=0
		while(counter<10):
			k_train_data = np.delete(train_data,slice(counter*height, (counter+1)*height),0)
			k_verif_data = train_data[counter*height:(counter+1)*height,:]
			
			x_train, y_train = getXandY(k_train_data)

			w = ridgeReggresion(x_train,y_train,lamda)
		
			x_verif, y_verif_true=getXandY(k_verif_data)
			
			y_verif_estimated=x_verif.dot(w)
			
			error = (error*counter+ np.linalg.norm(y_verif_estimated - y_verif_true)/x_verif.shape[0])/(counter+1)
			
			counter=counter+1

		errors[i] = error
		i=i+1

	my_lamda = params[np.argmin(errors)]
	x,y = getXandY(train_data)

	print(my_lamda)

	w=ridgeReggresion(x,y,my_lamda)

	np.savetxt(weightfile_txt,w)
	
	testDataOutput(testfile_csv,outputfile_txt,w)
if(mode=='c'):
	trainfile_csv = arguments[0]
	testfile_csv = arguments[1]

	outputfile_txt = arguments[2]

	train_data=np.array(genfromtxt(trainfile_csv,delimiter=','))
	params_big = np.array([0.00263])
	params_small = np.array([0])

	#params_big = np.array([0.0001,0.00025,0.0005,0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
	#s params_small = np.array([0.00001,0.000025,0.00005,0.000075,0.0001,0.00025,0.0005,0.001, 0.0025, 0.005, 0.01])
	k=10
	my_lamda=kFoldLasso(train_data,params_big,k)

	x_train, y_train = getXandY(train_data)

	model = linear_model.LassoLars(alpha=my_lamda)
	model.fit(x_train,y_train)
	
	w=model.coef_
	nonzero_cols = np.nonzero(w)
	print(nonzero_cols)
	x_train_reduced = getNonZeroCols(x_train,nonzero_cols)
	print(x_train_reduced.shape)
	print("------------------------------------")
	#apply feature creation
	x_train_reduced = createFeatures(x_train_reduced)

	my_lamda_reduced = kFoldLasso(x_train_reduced,params_small,k)
	model=linear_model.LassoLars(alpha=my_lamda_reduced)
	model.fit(x_train_reduced,y_train)
	
	x_test=np.array(genfromtxt(testfile_csv,delimiter=','))
	x_test=np.c_[np.ones(x_test.shape[0]), x_test]

	x_test=getNonZeroCols(x_test,nonzero_cols)
	print(x_test.shape)
	print("------------------------------------")
	x_test=createFeatures(x_test)
    
	output = model.predict(x_test)
	output = output.clip(min=0)
	np.savetxt(outputfile_txt, output, newline="\n")