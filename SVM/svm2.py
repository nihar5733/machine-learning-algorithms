import sys
import csv
import math
import numpy as np


def miniBatchPegasos(x,y,lr,k,T):
	w=np.zeros(x.shape[0])
	b=0

	for t in range(1,T):
		At=[]
		for r in np.random.randint(0,x.shape[0],size=k):
			if y[r]*(np.dot(x[r,:],w)+b) < 1: At.append(r)

		ct=1/(lr*t)
		s=np.zeros(w.shape[0])
		sb=0
		for i in At:
			s=s+y[i]*x[i,:]
			sb=sb+y[i]

		w=(1-ct*lr)*w + (lr/k)*s
		b=(1-ct*lr)*b + (lr/k)*sb

		projection=1/(np.sqrt(lr)*np.linalg.norm(w))
		if projection<1: w=projection*w

	return w,b

def train(x,y,lr,k,T):
	weights=[]
	bias=[]
	pos_one_class=[]
	neg_one_class=[]
	for i in range(0,10):
		for j in range(i+1,10):
			xt=np.r_[x[y==i,:],x[y==j,:]]
			yt=np.r_[y[y==i,:],y[y==j,:]]
			yt[yt==i]=1
			yt[yt==j]=-1

			w,b=miniBatchPegasos(xt,yt,lr,k,T)

			weights.append(w)
			bias.append(b)
			pos_one_class.append(i)
			neg_one_class.append(j)

	return weights, bias, pos_one_class, neg_one_class


def predict(x,weights,bias,pos_one_class,neg_one_class):
	y=[]
	for i in range(len(weights)):
		yp=np.dot(x,weights[i])+bias[i]
		yp[yp>=0]=pos_one_class[i]
		yp[yp<0]=neg_one_class[i]

		y.append(yp)

	y=np.array(y).T
	finalY=[]
	for j in range(y.shape[0]):
		(v,counts) = np.unique(y[j,:],return_counts=True)
		ind=np.argmax(counts)
		finalY.append(v[ind])

	return finalY


trainﬁle_csv = sys.argv[1]
testﬁle_csv = sys.argv[2]
testpred_txt = sys.argv[3]

trainﬁle = np.loadtxt(trainfile_csv,delimiter=',')
testfile = np.loadtxt(testﬁle_csv,delimiter=',')
print('-------------')

train_x = trainﬁle[:,:-1]
train_y = trainﬁle[:,-1].astype(int)
test_x = testfile[:,:-1]

lr=10000
k=1
T=train_x.shape[0]

weights, bias, pos_one_class, neg_one_class = train(train_x,train_y,lr,k,T)

test_y = predict(test_x,weights,bias,pos_one_class,neg_one_class)

print(test_y[0:10])

np.savetxt(testpred_txt,test_y, fmt='%d')


