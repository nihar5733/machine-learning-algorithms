import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

my_x_map = {
	
	#--------age: continuous---------

	#workclass
	'Private':0, 
	'Self-emp-not-inc':1, 
	'Self-emp-inc':2, 
	'Federal-gov':3, 
	'Local-gov':4, 
	'State-gov':5, 
	'Without-pay':6, 
	'Never-worked':7,

	#--------fnlwgt: continuous--------

	#education
	'Bachelors':8, 
	'Some-college':9, 
	'11th':10, 
	'HS-grad':11, 
	'Prof-school':12, 
	'Assoc-acdm':13, 
	'Assoc-voc':14, 
	'9th':15, 
	'7th-8th':16, 
	'12th':17, 
	'Masters':18, 
	'1st-4th':19, 
	'10th':20, 
	'Doctorate':21, 
	'5th-6th':22, 
	'Preschool':23,

	#-------education-num: continuous--------

	#marital-status
	'Married-civ-spouse':24, 
	'Divorced':25, 
	'Never-married':26, 
	'Separated':27, 
	'Widowed':28, 
	'Married-spouse-absent':29, 
	'Married-AF-spouse':30,

	#occupation
	'Tech-support':31, 
	'Craft-repair':32, 
	'Other-service':33, 
	'Sales':34, 
	'Exec-managerial':35, 
	'Prof-specialty':36, 
	'Handlers-cleaners':37, 
	'Machine-op-inspct':38, 
	'Adm-clerical':39, 
	'Farming-fishing':40, 
	'Transport-moving':41, 
	'Priv-house-serv':42, 
	'Protective-serv':43, 
	'Armed-Forces':44,

	#relationship
	'Wife':45, 
	'Own-child':46, 
	'Husband':47, 
	'Not-in-family':48, 
	'Other-relative':49, 
	'Unmarried': 50,

	#race 
	'White':51, 
	'Asian-Pac-Islander':52, 
	'Amer-Indian-Eskimo':53, 
	'Other':54, 
	'Black':55,

	#sex 
	'Female':56, 
	'Male':57,

	#--------capital-gain: continuous---------
	#--------capital-loss: continuous---------
	#-------hours-per-week: continuous--------

	#native-country
	'United-States':58, 
	'Cambodia':59, 
	'England':60, 
	'Puerto-Rico':61, 
	'Canada':62, 
	'Germany':63, 
	'Outlying-US(Guam-USVI-etc)':64, 
	'India':65, 
	'Japan':66, 
	'Greece':67, 
	'South':68, 
	'China':69, 
	'Cuba':70, 
	'Iran':71, 
	'Honduras':72, 
	'Philippines':73, 
	'Italy':74, 
	'Poland':75, 
	'Jamaica':76, 
	'Vietnam':77, 
	'Mexico':78, 
	'Portugal':79, 
	'Ireland':80, 
	'France':81, 
	'Dominican-Republic':82, 
	'Laos':83, 
	'Ecuador':84, 
	'Taiwan':85, 
	'Haiti':86, 
	'Columbia':87, 
	'Hungary':88, 
	'Guatemala':89, 
	'Nicaragua':90, 
	'Scotland':91, 
	'Thailand':92, 
	'Yugoslavia':93, 
	'El-Salvador':94, 
	'Trinadad&Tobago':95, 
	'Peru':96, 
	'Hong':97, 
	'Holand-Netherlands':98
}

def oneHotEncode(xraw):
	x = np.zeros([xraw.shape[0],105])
	x[:,99] = xraw[:,0]
	x[:,100] = xraw[:,2]
	x[:,101] = xraw[:,4]
	x[:,102] = xraw[:,10]
	x[:,103] = xraw[:,11]
	x[:,104] = xraw[:,12]

	for i in range(xraw.shape[0]):
		for j in range(xraw.shape[1]):
			if(j==0 or j==2 or j==4 or j==10 or j==11 or j==12): continue
			else: x[i,my_x_map[(str(xraw[i,j])).strip()]]=1

	return x

class Node:
	def __init__(self,xj,v,l,r):
		self.feature=xj
		self.value=v
		self.left=l
		self.right=r
		self.px=[]
		self.py=[]

	def check(self,x): #True for Left
		return x[self.feature]<=self.value

	def appendPxPy(self,x,y):
		self.px.append(x)
		self.py.append(y)
		return self
	
	def cleanNode(self):
	    self.px=[]
	    self.py=[]
	    return self
	


def getEntropy(y):
	if(y.shape[0]==0): return 0
	p = np.count_nonzero(y==1)/y.shape[0]
	q = 1-p
	if(p==0 or q==0): return 0
	return -p*math.log(p) - q*math.log(q)


def split(x,y):
	entropy = getEntropy(y)
	maxgain = -1
	feature = -1
	value = 0
	for i in range(105):
		xi=x[:,i]
		if(i<=98):
			l = y[xi==0]
			r = y[xi==1]

			gain = entropy - (l.shape[0]/y.shape[0])*getEntropy(l) - (r.shape[0]/y.shape[0])*getEntropy(r)
			if(gain>maxgain):
				maxgain=gain
				feature=i
				value=0
		else:
			#-------mean splitting--------
			v=np.median(xi)
			l=y[xi<=v]
			r=y[xi>v]
			gain = entropy - (l.shape[0]/y.shape[0])*getEntropy(l) - (r.shape[0]/y.shape[0])*getEntropy(r)
			if(gain>maxgain):
				maxgain=gain
				feature=i
				value=v



			#-------threshold based dplitting-----------
			# y=y[xi.argsort()]
			# xi=xi[xi.argsort()]
			# old = y[0]
			# for j in range(y.shape[0]):
			# 	new=y[j]
			# 	if(new==old):continue
			# 	else:
			# 		old=new
			# 		v=(xi[j-1]+xi[j])/2
			# 		l=y[0:j]
			# 		r=y[j:]
			# 		gain = entropy - (l.shape[0]/y.shape[0])*getEntropy(l) - (r.shape[0]/y.shape[0])*getEntropy(r)
			# 		if(gain>maxgain):
			# 			maxgain=gain
			# 			feature=i
			# 			value=v
	return feature,value



def growTrees(x,y,count):
	if(count<=0):
		v=0
		ones = np.count_nonzero(y==1)
		if(ones>y.shape[0]/2):v=1
		return Node(None,v,None,None)



	if(y.shape[0]==0):
		return None
	elif(np.all(y==0)): return Node(None,0,None,None)
	elif(np.all(y==1)): return Node(None,1,None,None)
	else:
		xj,v = split(x,y)

		x0 = x[x[:,xj] <= v]
		y0 = y[x[:,xj] <= v]
		x1 = x[x[:,xj] > v]
		y1 = y[x[:,xj] > v]
		if (y0.shape[0]==0):
			v=0
			ones = np.count_nonzero(y1==1)
			if(ones>y1.shape[0]/2):v=1
			return Node(None,v,None,None)
		elif (y1.shape[0]==0):
			v=0
			ones = np.count_nonzero(y0==1)
			if(ones>y0.shape[0]/2):v=1
			return Node(None,v,None,None)
		return Node(xj,v,growTrees(x0,y0,count-2),growTrees(x1,y1,count-2))


def predict(root,x):
	y=np.zeros(x.shape[0])
	
	for i in range(x.shape[0]):
		flag=True
		ptr=root
		v=0
		while (flag):
			v=ptr.value
			if(ptr.feature != None):
				if(ptr.check(x[i,:])):ptr=ptr.left
				else:ptr=ptr.right
			else:flag=False
		y[i]=v

	return y
	

def cleanAfterPruning(node):
    if (node.feature == None): return node.cleanNode()
    else:
        node.left=cleanAfterPruning(node.left)
        node.right=cleanAfterPruning(node.right)
        node=node.cleanNode()
        
        return node


def prune(node):
	x=np.array(node.px)
	y=np.array(node.py)

	y_pred = predict(node,x)
	e = np.linalg.norm(y_pred-y)
	ep = np.count_nonzero(y==1)
	v=1
	if(ep<y.shape[0]/2):
		ep = y.shape[0]-ep
		v=0

	ep=math.sqrt(ep)

	if(ep<=e):
		return Node(None,v,None,None)
	
	return node


def postPruning(node):
	if(node.feature==None): return node
	else:
		node.left=postPruning(node.left)
		node.right=postPruning(node.right)
		node=prune(node)

		return node

def pruneMyTree(root,xv,yv):
	
	for i in range(xv.shape[0]):
		flag=True
		ptr=root
		while (flag):
			ptr=ptr.appendPxPy(xv[i,:],yv[i])
			if(ptr.feature != None):
				if(ptr.check(xv[i,:])):ptr=ptr.left
				else:ptr=ptr.right
			else:flag=False

	root=postPruning(root)
	# print(len(root.px))
	
	root=cleanAfterPruning(root)
	# print(len(root.px))
	return root



trainﬁle_csv = sys.argv[1]
validﬁle_csv = sys.argv[2]
testﬁle_csv = sys.argv[3]
validpred_txt = sys.argv[4]
testpred_txt = sys.argv[5]

trainﬁle = np.loadtxt(trainfile_csv,delimiter=',',dtype=object)
testfile = np.loadtxt(testﬁle_csv,delimiter=',',dtype=object)
validfile = np.loadtxt(validﬁle_csv,delimiter=',',dtype=object)

trainﬁle = trainﬁle[1:,:]
testfile = testfile[1:,:]
validfile = validfile[1:,:]

train_x = trainﬁle[:,:-1]
train_y = trainﬁle[:,-1].astype(int)
test_x = testfile[:,:-1]
valid_x = validfile[:,:-1]
valid_y = validfile[:,-1].astype(int)

train_x = oneHotEncode(train_x)
test_x = oneHotEncode(test_x)
valid_x = oneHotEncode(valid_x)

test_y=np.loadtxt('testpred.txt',dtype=int)


train_accu=[]
valid_accu=[]
test_accu=[]
xplot=[]
count=0
while (count<=10000):
	count=count+100
	print(count)

	root=growTrees(train_x,train_y,count)

	# y_valid_pred=predict(root,valid_x)
	# e0=np.linalg.norm(valid_y-y_valid_pred)
	# pcheck=True
	# while(pcheck):
	#     root=pruneMyTree(root,valid_x,valid_y)
	#     y_valid_pred=predict(root,valid_x)
	#     e1=np.linalg.norm(valid_y-y_valid_pred)
	#     # print(e0,e1)
	#     if(e1>=e0):pcheck=False
	#     e0=e1
	y_train_pred=predict(root,train_x)
	y_test_pred=predict(root,test_x)
	y_valid_pred=predict(root,valid_x)

	trac=np.sum(np.abs(y_train_pred - train_y)==0)/train_y.shape[0]
	tsac=np.sum(np.abs(y_test_pred - test_y)==0)/test_y.shape[0]
	vaac=np.sum(np.abs(y_valid_pred - valid_y)==0)/valid_y.shape[0]

	root=None

	train_accu.append(trac)
	test_accu.append(tsac)
	valid_accu.append(vaac)
	xplot.append(count)

plt.plot(xplot,train_accu)
plt.plot(xplot,test_accu)
plt.plot(xplot,valid_accu)
plt.legend(['Training accuracy', 'Test accuracy', 'Validation accuracy'])
plt.xlabel('Number of nodes')
plt.ylabel('Accuracy')
plt.show()

# np.savetxt(testpred_txt,y_test_pred, fmt='%d')
# np.savetxt(validpred_txt,y_valid_pred, fmt='%d')