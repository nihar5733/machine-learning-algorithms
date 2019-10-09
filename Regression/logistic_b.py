import sys
import csv
import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

my_x_map = {
	'usual0':0, 
	'pretentious0':1, 
	'great_pret0':2,
	'proper1':3, 
	'less_proper1':4, 
	'improper1':5, 
	'critical1':6, 
	'very_crit1':7, 
	'complete2':8, 
	'completed2':9, 
	'incomplete2':10, 
	'foster2':11, 
	'13':12, 
	'23':13, 
	'33':14, 
	'more3':15, 
	'convenient4':16, 
	'less_conv4':17, 
	'critical4':18, 
	'convenient5':19, 
	'inconv5':20, 
	'nonprob6':21, 
	'slightly_prob6':22, 
	'problematic6':23, 
	'recommended7':24, 
	'priority7':25, 
	'not_recom7':26
}

my_y_map = {
	'not_recom':0, 
	'recommend':1, 
	'very_recom':2, 
	'priority':3, 
	'spec_prior':4
}

my_reverse_y_map = {
	0:'not_recom', 
	1:'recommend', 
	2:'very_recom', 
	3:'priority', 
	4:'spec_prior'
}


def getXandY(dataset):
	y=dataset[:,-1]
	x=dataset[:,:-1]
	return x,y

def getYhat(x,w):
	mat = np.exp(np.dot(x,w))
	mat=(mat.T/sum(mat.T)).T
	
	return mat

def oneHotEncodeX(x):
	x_encoded = np.zeros([x.shape[0],27],dtype=float)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x_encoded[i,my_x_map[x[i,j]+str(j)]]=1
	return np.array(x_encoded)

def oneHotEncodeY(y):
	y_encoded = np.zeros([y.shape[0],5],dtype=float)
	for i in range(y.shape[0]):
		y_encoded[i,my_y_map[y[i]]]=1
	return y_encoded

def miniBatchGradientDescentFixedRate(x,y,learning_rate,max_itr,batch_size):
    w = np.zeros([x.shape[1],y.shape[1]],dtype=float)
    noOfBatches = math.floor(x.shape[0]/batch_size)
    itr=0
    while itr<max_itr:
        for i in range(noOfBatches):
            x_batch = x[i*batch_size:(i+1)*batch_size,:]
            y_batch = y[i*batch_size:(i+1)*batch_size,:]
            yhat=getYhat(x_batch,w)
            w=w+learning_rate*np.dot(x_batch.T,y_batch-yhat)/x_batch.shape[0]
        itr=itr+1
    
    return w
    

def miniBatchgradientDescentAdaptiveRate(x,y,seed_value,max_itr,batch_size):
    w = np.zeros([x.shape[1],y.shape[1]],dtype=float)
    noOfBatches = math.floor(x.shape[0]/batch_size)
    itr=0
    while itr<max_itr:
        itr=itr+1
        for i in range(noOfBatches):
            x_batch = x[i*batch_size:(i+1)*batch_size,:]
            y_batch = y[i*batch_size:(i+1)*batch_size,:]
            yhat=getYhat(x_batch,w)
            #-----------------------------------------DOUBT---------------------------------------sqrt(itr) or what?--------
            w=w+(seed_value/math.sqrt(itr))*np.dot(x_batch.T,y_batch-yhat)/x_batch.shape[0]
    return w

def negLikelihood(w,x,y):
	xw=np.dot(x,w)
	xwexp=np.exp(xw)
	mysum=0
	for i in range(x.shape[0]):
		j=np.argmax(y[i,:])
		mysum += xw[i,j]-np.log(sum(xwexp[i,:]))

	return -mysum/x.shape[0]


def gradientDescentBacktracking(x,y,learning_rate_start,alpha,beta,max_itr,batch_size):
    w = np.zeros([x.shape[1],y.shape[1]],dtype=float)
    noOfBatches = math.floor(x.shape[0]/batch_size)
    itr=0
    learning_rate=learning_rate_start
    while itr<max_itr:
        #learning_rate=learning_rate_start
        for i in range(noOfBatches):
            # learning_rate=learning_rate_start
            x_batch = x[i*batch_size:(i+1)*batch_size,:]
            y_batch = y[i*batch_size:(i+1)*batch_size,:]
            yhat=getYhat(x_batch,w)
            gradient=np.dot(x_batch.T,yhat-y_batch)/x.shape[0]
            gradient1x1=gradient.reshape(gradient.shape[0]*gradient.shape[1])
            while(negLikelihood(w-learning_rate*gradient,x_batch,y_batch) > negLikelihood(w,x_batch,y_batch) + alpha*learning_rate*np.dot(gradient1x1, gradient1x1)):
                learning_rate=beta*learning_rate
                
            w=w-learning_rate*gradient
        itr=itr+1
    
    return w

trainﬁle_csv = sys.argv[1]
testﬁle_csv = sys.argv[2]
param_txt = sys.argv[3]
outputﬁle_csv = sys.argv[4]
weightﬁle_csv = sys.argv[5]


trainﬁle = np.genfromtxt(trainfile_csv,delimiter=',',dtype=str)
# params_file = open(param_txt,"r")
# params_str = params_file.readlines()

strategy = 2
# max_itr = int(params_str[2])
# batch_size = int(params_str[3])
# rate=params_str[1].split(',')
# rate=np.array(rate,dtype=float)

# params_file.close()


trainﬁle_x,trainﬁle_y = getXandY(trainﬁle)

x_encoded = oneHotEncodeX(trainﬁle_x)
x_encoded=np.c_[np.ones([x_encoded.shape[0],1]),x_encoded]

y_encoded = oneHotEncodeY(trainﬁle_y)
if(strategy==1):
	max_itr=10000
	learning_rate=0.1
	batch_size=100
	# batch_sizes=[10,50,100,150,200,250,300,400,500,1000,2000,3000]
	# learning_rates = [0.01, 0.05,0.1,0.3, 0.5, 1]
	# n=x_encoded.shape[0]
	# for i in learning_rates:
	# 	Ls=np.zeros(len(batch_sizes))
	# 	fpos=np.zeros(len(batch_sizes))
	# 	ptr=0
	# 	for j in batch_sizes:
	w0=miniBatchGradientDescentFixedRate(x_encoded,y_encoded,learning_rate,max_itr,batch_size)
	# 		Ls[ptr]=-negLikelihood(w0,x_encoded,y_encoded)
	# 		fpos[ptr]=math.floor(n/j)*max_itr*(560*j+10*j+450)
	# 		ptr+=1
	# 		print(j)
	# 	plt.plot(fpos,Ls)
	# 	print('____________________________')
	# 	print(i)
	# plt.xlabel('No of floating point operations')
	# plt.ylabel('Likelihood function, L(w;x,y)')
	# plt.title('Constant rate mini batch Gradient descent:\nL(w;x,y) wrt floating point operations\n For different learning rates with varying batch sizes')
	# plt.legend(['0.01','0.05','0.1','0.3', '0.5', '1'], loc='lower right')
	# plt.show()



elif(strategy==2):
	max_itr=10000
	learning_rate=10
	batch_size=100
	# batch_sizes=[10,50,100,150,200,250,300,400,500,1000,2000,3000]
	# learning_rates = [0.5, 1,5,10, 15, 20]
	# n=x_encoded.shape[0]
	# for i in learning_rates:
	# 	print('here')
	# 	Ls=np.zeros(len(batch_sizes))
	# 	fpos=np.zeros(len(batch_sizes))
	# 	ptr=0
	# 	for j in batch_sizes:
	w0=miniBatchgradientDescentAdaptiveRate(x_encoded,y_encoded,learning_rate,max_itr,batch_size)
	# 		Ls[ptr]=-negLikelihood(w0,x_encoded,y_encoded)
	# 		fpos[ptr]=math.floor(n/j)*max_itr*(560*j+10*j+450)
	# 		ptr+=1
	# 		print(j)
	# 	plt.plot(fpos,Ls)
	# 	print(i)
	# 	print('____________________________')
	# plt.xlabel('No of floating point operations')
	# plt.ylabel('log Likelihood function, L(w;x,y)')
	# plt.title('Adaptive rate mini batch Gradient descent:\nL(w;x,y) wrt floating point operations\n For different learning rates with varying batch sizes')
	# plt.legend(['0.5','1','5','10', '15', '20'], loc='lower right')
	# plt.show()
    # seed_value = rate[0]
    # w0=miniBatchgradientDescentAdaptiveRate(x_encoded,y_encoded,seed_value,max_itr,batch_size)
elif(strategy==3):
    learning_rate_start=rate[0]
    alpha=rate[1]
    beta=rate[2]
    w0=gradientDescentBacktracking(x,y,learning_rate_start,alpha,beta,max_itr,batch_size)


#-----------------------check-----------------------

my_yhat = getYhat(x_encoded,w0)
my_yhat = (my_yhat == my_yhat.max(axis=1)[:,None]).astype(int)
e=np.linalg.norm(y_encoded-my_yhat)
confusion = np.dot(y_encoded.T,my_yhat)

print('Confusion matrix:');print(confusion)
rowsum=np.sum(confusion,axis=1)
colsum=np.sum(confusion,axis=0)

precision=(confusion.diagonal())/rowsum
precision[np.isnan(precision)]=0
print('precision:');print(precision)

recall=(confusion.diagonal())/colsum
recall[np.isnan(recall)]=0
print('recall:');print(recall)

F1 = 2*(1/((1/precision)+(1/recall)))
print('F1:');print(F1)

macroF1 = 2*(1/((5/np.sum(precision))+(5/np.sum(recall))))
print('Macro F1:');print(macroF1)

microF1 = np.trace(confusion)/np.sum(confusion)
print('Micro F1:');print(microF1)

#-----------------------check-----------------------


testﬁle = np.genfromtxt(testﬁle_csv,delimiter=',',dtype=str)

test_x = oneHotEncodeX(testﬁle)
test_x=np.c_[np.ones([test_x.shape[0],1]),test_x]

y_predicted = np.dot(test_x,w0)

y_decoded = testfile[:,0]
for decode in range(y_decoded.shape[0]):
	y_decoded[decode] = my_reverse_y_map[np.argmax(y_predicted[decode])]

np.savetxt(weightﬁle_csv,w0,delimiter=',')
np.savetxt(outputﬁle_csv,y_decoded,delimiter=',',fmt="%s")