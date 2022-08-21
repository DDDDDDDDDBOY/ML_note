import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math

def create_data():
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   df['label'] = iris.target
   df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
   data = np.array(df.iloc[:100, :])
   # print(data)
   return data[:,:-1], data[:,-1]
X, y = create_data()

def navieBayes(X,y):
    # first 25 items and last 25 items for training
    prior1,prior2 = priorP(y)
    x1 = X[:25]
    x2 = X[75:]
    mean1,stdev1 = condP(x1)
    mean2,stdev2 = condP(x2)
    # instance for test
    xp1 = X[42]
    xp2 = X[59]
    res1 = predict(xp1,prior1,prior2,mean1,mean2,stdev1,stdev2)
    res2 = predict(xp2,prior1,prior2,mean1,mean2,stdev1,stdev2)
    print(f'prediction for instance xp1 is {res1}\n prediction for instance xp2 is {res2}')

# calculate the prior probability
def priorP(data):
    # a = class 1 b = class 2
    a = 0
    b = 0
    for i in data:
        if i == 0:
            a += 1
        else:
            b += 1
    return a/len(data) , b/len(data)
# calculate the conditional probability(likelihood)by using gaussian distribution
def condP(data):
    #calculate mean and stdev of every feature
    mean = []
    stdev = []
    a = 0
    for i in range (len(data[0])):
        for l in data:
            a += l[i]
        mean.append(a/len(data))
        a = 0
    for i in range (len(data[0])):
        for l in data:
            a += (l[i]-mean[i])**2
        stdev.append(math.sqrt(a/len(data)))
        a = 0
    return mean,stdev

def postP():
    pass

# predict a new instance
def predict(data,prior1,prior2,mean1,mean2,stdev1,stdev2):
    p1 = prior1
    p2 = prior2
    k=0
    print(p1,p2,mean1,mean2,stdev1,stdev2)
    for i in data:
        p1 = p1*(1/(stdev1[k]*math.sqrt(2*math.pi))*math.exp(-(i-mean1[k])**2)/2*(stdev1[k])**2)
        k += 1
    k=0
    for i in data:
        p2 = p2*(1/(stdev2[k]*math.sqrt(2*math.pi))*math.exp(-(i-mean2[k])**2)/2*(stdev2[k])**2)
        k += 1
    #prevent underflow
    p1 = math.log(p1)
    p2 = math.log(p2)
    print(p1,p2)
    if p1>p2:
        return 0
    else:
        return 1

navieBayes(X,y)
