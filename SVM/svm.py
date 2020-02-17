import sys
import csv
import math
import numpy as np


def miniBatchPegasos(x,y,lr,batch_size):
    w=np.zeros(x.shape[1])
    b=0
    for t in range(x.shape[0]):
        xt=[]
        yt=[]
        for r in np.random.randint(0,x.shape[0],size=batch_size):
            if(y[r]*(np.dot(x[r,:],w)+b)<1):
                xt.append(x[r,:])
                yt.append(y[r])
        xt=np.array(xt)
        yt=np.array(yt)
        s=np.zeros(w.shape[0])
        sb=0
        for p in range(len(yt)):
           s=s+xt[p,:]*yt[p]
           sb=sb+yt[p]
        
        c_t = 1/(lr*(t+1))
        
        w=(1-c_t*lr)*w + (c_t/batch_size)*s
        b=(1-c_t*lr)*b + (c_t/batch_size)*sb
        
        f=1/(np.sqrt(lr)*np.linalg.norm(w))
        if(f < 1):
            w=f*w
    
    return w,b

# 		check=np.multiply(yt,np.dot(xt,w))
# 		i=np.argwhere(check<1)
# 		i=np.squeeze(np.asarray(i))
# 		if i==None: i=[]
# 		s=0
# 		for p in range(len(i)):
# 		    s=s+xt[i[p]]*yt[i[p]]
# # 		s=np.sum(np.multiply(xt[i].T,yt[i]).T,axis=0)

def predict(x,weights,bs,pos_one,neg_one):
	y=[]
	for i in range(len(weights)):
		yp=np.dot(x,weights[i]) + bs[i]
		yp[yp>=0]=pos_one[i]
		yp[yp<0]=neg_one[i]
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


batch_size=1
lr=100000

pos_one=[]
neg_one=[]
weights=[]
bs=[]

for i in range(10):
	for j in range(i+1,10):
		arg_i=np.argwhere(train_y==i)
		arg_i=np.squeeze(np.asarray(arg_i))
		arg_j=np.argwhere(train_y==j)
		arg_j=np.squeeze(np.asarray(arg_j))
		x=np.r_[train_x[arg_i,:],train_x[arg_j,:]]
		y=np.r_[train_y[arg_i],train_y[arg_j]]
		#x=train_x[np.append(arg_i,arg_j)]
		#y=train_y[np.append(arg_i,arg_j)]
		y[y==i]=1
		y[y==j]=-1
		
		print(x.shape,y.shape)

		w,b=miniBatchPegasos(x,y,lr,batch_size)
		print(i,j)

		pos_one.append(i)
		neg_one.append(j)
		weights.append(w)
		bs.append(b)


test_y=predict(x,weights,bs,pos_one,neg_one)
print(test_y[0:10])

np.savetxt(testpred_txt,test_y, fmt='%d')