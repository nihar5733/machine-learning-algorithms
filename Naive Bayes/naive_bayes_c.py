import sys
import csv
import math
import numpy as np
from collections import defaultdict
from nltk.stem.porter import *
from nltk import bigrams


stop_words = {'those', 'who', 'your', 'were', 'into', 'no', 'hadn', 'will', 'which', 'you', 'him', 'me', "you'll", 'should', 'her', "weren't", 'needn', 'to', 't', 'through', 'yours', 'had', 'with', 'm', 'out', 'herself', 'up', "wasn't", 'again', 're', "it's", 'this', 'and', "shan't", "that'll", 'whom', 'over', 'is', 'more', 'against', 'ma', "shouldn't", 'when', 'myself', 'after', 'between', 'here', "won't", 'd', 'yourself', 'such', 'haven', 'isn', 'having', 'y', 'ourselves', 'not', 'it', 'below', 'he', 'while', 'now', "aren't", 'until', 'did', 'i', 'too', 'our', 'an', 'the', 'theirs', 'was', 'where', 'from', "hadn't", 'then', 'weren', 'each', 'a', 'off', 'why', 'once', 'didn', 'aren', 'than', 'are', 'am', 'during', 'above', 'them', 'at', 'don', 'they', 'if', 'been', "needn't", 'won', "haven't", 'but', 'so', 've', 'just', "you've", 'by', "should've", "you're", 'that', 'yourselves', 'do', 'same', "don't", 'hers', 'both', 'does', 'shan', 'or', 'further', "isn't", 'itself', 'on', 'only', 'own', 'how', 'for', 'o', 'being', 'ours', "doesn't", 'of', "she's", 'wouldn', 'have', 'their', 'himself', 'doing', 'his', 'has', "hasn't", 'all', 'what', 'any', 'few', "didn't", 'before', 'shouldn', 's', 'doesn', 'down', 'very', 'these', 'wasn', 'themselves', 'be', 'she', 'most', 'ain', 'couldn', "couldn't", 'as', 'll', 'in', 'its', 'hasn', 'under', "mustn't", "wouldn't", "you'd", 'can', 'other', "mightn't", 'my', 'there', 'some', 'about', 'mustn', 'mightn', 'nor', 'we', 'because'}
stemmer = PorterStemmer()

def trainNaiveBayes(reviews,sentiment,alpha=1):
	posWords=defaultdict(int)
	negWords=defaultdict(int)
	
	vocabulary=set()
	
	for r,s in zip(reviews,sentiment):
		if s=='positive':
			for w in r: posWords[w]+=1; vocabulary.add(w)
		if s=='negative':
			for w in r: negWords[w]+=1; vocabulary.add(w)
	
	totPosCount=sum(posWords.values())
	totNegCount=sum(negWords.values())
	
	posProb=totPosCount/(totPosCount+totNegCount)
	negProb=1-posProb
	for w in vocabulary:
		posWords[w]=np.log((posWords[w]+alpha)/(totPosCount+alpha*2))
		negWords[w]=np.log((negWords[w]+alpha)/(totNegCount+alpha*2))

	loglikelihood={0:negWords, 1:posWords, 2:np.log(alpha/(totNegCount+alpha*2)), 3:np.log(alpha/(totPosCount+alpha*2))}
	
	logprior={0:negProb, 1:posProb}
	
	return loglikelihood, logprior, vocabulary


def predict(review,loglikelihood,logprior,vocabulary):
	sums=[logprior[0], logprior[1]]
	
	for i in [0,1]:
		for w in review:
			if w in vocabulary:
				sums[i] += loglikelihood[i][w]
			else:
			    sums[i] += loglikelihood[i+2]

	#print(sums)
	return np.array(sums)


trainﬁle_csv = sys.argv[1]
testﬁle_csv = sys.argv[2]
outputﬁle_txt = sys.argv[3]

punctuation = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'


trainﬁle=[]
with open(trainﬁle_csv, 'r',encoding="utf8") as f:
	reader = csv.reader(f)
	for row in reader:
		if (row[0] != 'review'):
		    row[0]=row[0].lower()
		    for char in punctuation: row[0]=row[0].replace(char, ' ')
		    row[0]=row[0].split()
		    row[0]=[w for w in row[0] if not w in stop_words]
		    #row[0]=[stemmer.stem(w) for w in row[0]]
		    bi = list(bigrams(row[0]))
		    bi =[''.join(w+' ' for w in b).strip() for b in bi]
		    
		    row[0]=row[0]+bi 
		    
		    trainﬁle.append(row)

testfile=[]
with open(testﬁle_csv, 'r',encoding="utf8") as f:
	reader = csv.reader(f)
	for row in reader:
		if (row[0] != 'review'): 
		    row[0]=row[0].lower()
		    for char in punctuation: row[0]=row[0].replace(char, ' ')
		    row[0]=row[0].split()
		    row[0]=[w for w in row[0] if not w in stop_words]
		    #row[0]=[stemmer.stem(w) for w in row[0]]
		    bi = list(bigrams(row[0]))
		    bi =[''.join(w+' ' for w in b).strip() for b in bi]
		    
		    row[0]=row[0]+bi
		    
		    testfile.append(row)

print(len(testfile))

reviews=[row[0] for row in trainfile]
sentiment=[row[1] for row in trainfile]

loglikelihood, logprior, V = trainNaiveBayes(reviews,sentiment,1)

print(len(V))

sentiment_predicted=np.zeros(len(testfile))
for i in range(len(testfile)):
	sentiment_predicted[i]=np.argmax(predict(testfile[i][0],loglikelihood,logprior,V))

np.savetxt(outputﬁle_txt,sentiment_predicted, fmt='%d')

