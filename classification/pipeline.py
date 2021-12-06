from pyspark.ml.feature import *
from pyspark.ml import Pipeline
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
from sklearn import linear_model
from pyspark.ml import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cluster import MiniBatchKMeans
counter_pareg=0
counter_sgd=0
counter_nb=0
mbk_counter=0

PA_I_online = PassiveAggressiveClassifier(loss='hinge')
clf_sgd=linear_model.SGDClassifier()
clf=GaussianNB()
mbk = MiniBatchKMeans(n_clusters=2, init='k-means++',batch_size=500) # Batch size needs to be varied



def hashvectorizer(train):
    X=train.select('message').collect()
    X=[i['message'] for i in X]
    vectorizer = HashingVectorizer(n_features=100)
    X = vectorizer.fit_transform(X)
    y=train.select('label').collect()
    y=np.array([i[0] for i in np.array(y)])
    return X,y





def ClusteringGo(train):
    global mbk
    global mbk_counter
    mbk_counter+=1
    X,y=hashvectorizer(train)
    mbk.partial_fit(X)
    if mbk_counter == 60: #counter_sgd is 30 when batch size is 1000 and 303 when batch size is 100 and 60 when the batch size is 500
        #save model into pickle file 
        with open('MiniBatchKmeans_500', 'wb') as files:
             pickle.dump(mbk, files)
    MiniBatchKMeans(n_clusters=2, init='k-means++',batch_size=500) # Batch size needs to be varied



def pareg(train):
    global counter_pareg
    global PA_I_online
    counter_pareg+=1
    X,y=hashvectorizer(train)
    no_of_classes=np.unique(y)
    PA_I_online.partial_fit(X,y,no_of_classes)
    if counter_pareg ==30: 
        #save model into pickle file 
        print("there")
        with open('PAREGmodel', 'wb') as files:
            pickle.dump(PA_I_online, files)
        return
    PassiveAggressiveClassifier(loss='hinge')
    


def SgdClassifierGoTest(train):
    global clf_sgd
    global counter_sgd
    counter_sgd+=1
    X,y=hashvectorizer(train)
    clf_sgd.partial_fit(X.toarray(), y, classes = np.unique(y))
    if counter_sgd == 30: #counter_sgd is 30 when batch size is 1000 and 303 when batch size is 100 and 60 when the batch size is 500
        #save model into pickle file 
        
        with open('SGDmodel', 'wb') as files: 
            pickle.dump(clf_sgd, files)
        return
    linear_model.SGDClassifier()


def naiveBayesClassifierGo(train):    
    global clf
    global counter_nb
    counter_nb+=1
    X,y=hashvectorizer(train)
    clf.partial_fit(X.toarray(), y, classes = np.unique(y))
    if counter_nb ==30:   #counter_nb is 30 when batch size is 1000 and 303 when batch size is 100 and 60 when the batch size is 500
        #save model into pickle file 
        with open('NBmodel', 'wb') as files:
            pickle.dump(clf, files)
        return
    GaussianNB()
    
    
    
def sender(df):
    stages = []
# 1. Convert the labels to numerical values using binariser
    indexer = StringIndexer(inputCol="spam/ham", outputCol="label")
    stages += [indexer]
    
    [print('\n', stage) for stage in stages]
    pipeline = Pipeline(stages=stages)
    data = pipeline.fit(df).transform(df)
    print(data)
     
    #naiveBayesClassifierGo(data)
    #SgdClassifierGoTest(data)     #when to call sgd classifier 
    #ClusteringGo(data)
    pareg(data)

