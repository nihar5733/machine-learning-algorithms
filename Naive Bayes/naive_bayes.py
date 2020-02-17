import sys
import csv
import math
import string
import numpy as np
from collections import defaultdict


def trainNaiveBayes(reviews,sentiment,alpha=1):
	posWords=defaultdict(int)
	negWords=defaultdict(int)
	
	vocabulary=set()
	
	for r,s in zip(reviews,sentiment):
		if s=='positive':
			for w in r.split(" "): posWords[w]+=1; vocabulary.add(w)
		if s=='negative':
			for w in r.split(" "): negWords[w]+=1; vocabulary.add(w)
	
	totPosCount=sum(posWords.values())
	totNegCount=sum(negWords.values())
	
	posProb=totPosCount/(totPosCount+totNegCount)
	negProb=1-posProb
	for w in vocabulary:
		posWords[w]=np.log((posWords[w]+alpha)/(totPosCount+alpha*2))
		negWords[w]=np.log((negWords[w]+alpha)/(totNegCount+alpha*2))

	loglikelihood={0:negWords, 1:posWords}
	
	logprior={0:negProb, 1:posProb}
	
	return loglikelihood, logprior, vocabulary


def predict(review,loglikelihood,logprior,vocabulary):
	sums=[logprior[0], logprior[1]]
	
	for i in [0,1]:
		for w in review.split(" "):
			if w in vocabulary:
				sums[i] += loglikelihood[i][w]

	#print(sums)
	return np.array(sums)


trainﬁle_csv = sys.argv[1]
testﬁle_csv = sys.argv[2]
outputﬁle_txt = sys.argv[3]


trainﬁle=[]
with open(trainﬁle_csv, 'r',encoding="utf8") as f:
	reader = csv.reader(f)
	for row in reader:
		row[0]=row[0].lower()
		#for char in string.punctuation: row[0]=row[0].replace(char, ' ')
		trainﬁle.append(row)

testfile=[]
with open(testﬁle_csv, 'r',encoding="utf8") as f:
	reader = csv.reader(f)
	for row in reader:
		row[0]=row[0].lower()
		#for char in string.punctuation: row[0]=row[0].replace(char, ' ')
		testfile.append(row)

trainﬁle = np.array(trainﬁle)
# testfile = np.array(testfile)

print(len(testfile))

trainﬁle=trainﬁle[1:,:]
testfile=testfile[1:][:]

reviews=trainﬁle[:,0]
sentiment=trainﬁle[:,1]

loglikelihood, logprior, V = trainNaiveBayes(reviews,sentiment,1)

print(len(V))

sentiment_predicted=np.zeros(len(testfile))
for i in range(len(testfile)):
	sentiment_predicted[i]=np.argmax(predict(testfile[i][0],loglikelihood,logprior,V))

np.savetxt(outputﬁle_txt,sentiment_predicted, fmt='%d')

