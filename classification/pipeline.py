from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
from sklearn import linear_model
from pyspark.ml import Pipeline
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover
# Use English stemmer.

#global variables for naivebayes
counter_nb=0
clf=GaussianNB()


#global variables for SGDClassifier
clf_sgd=linear_model.SGDClassifier()
counter_sgd=0


#global variables for PAREGS
PA_I_online = PassiveAggressiveClassifier(loss='hinge')
counter_pareg=0



def pareg(train):
    
    global counter_pareg
    global PA_I_online
    counter_pareg+=1

    X=train.select('message').collect()
    X=[i['message'] for i in X]
    vectorizer = HashingVectorizer(n_features=100000)
    X = vectorizer.fit_transform(X)
    y=train.select('label').collect()
    y=np.array([i[0] for i in np.array(y)])
    no_of_classes=np.unique(y)
    PA_I_online.partial_fit(X,y,no_of_classes)



    if counter_pareg ==303: 
        #save model into pickle file 
        with open('PAREGmodel', 'wb') as files:
             pickle.dump(PA_I_online, files)
        return
    PassiveAggressiveClassifier(loss='hinge')


def SgdClassifierGoTest(train):
    global clf_sgd
    global counter_sgd
    counter_sgd+=1
    
    X=train.select('message').collect()
    X=[i['message'] for i in X]
    vectorizer = HashingVectorizer(n_features=100000)
    X = vectorizer.fit_transform(X)
    y=train.select('label').collect()
    y=np.array([i[0] for i in np.array(y)])
    clf_sgd.partial_fit(X.toarray(), y, classes = np.unique(y))
  


    if counter_sgd == 303: #counter_sgd is 30 when batch size is 1000 and 303 when batch size is 100
        #save model into pickle file 
        with open('SGDmodel_1000', 'wb') as files:
             pickle.dump(clf_sgd, files)
        return
    linear_model.SGDClassifier()


def naiveBayesClassifierGo(train):
        
    global clf
    global counter_nb
    counter_nb+=1


    X=train.select('message').collect()
    X=[i['message'] for i in X]
    vectorizer = HashingVectorizer(n_features=100000)
    X = vectorizer.fit_transform(X)
    y=train.select('label').collect()
    y=np.array([i[0] for i in np.array(y)])
    clf.partial_fit(X.toarray(), y, classes = np.unique(y))



    if counter_nb ==303: 
        #save model into pickle file 
        with open('NBmodelrevise', 'wb') as files:
             pickle.dump(clf, files)
        return
    GaussianNB()
    
    



def sender(df):
    stages = []
    
    # 1. clean data and tokenize sentences using RegexTokenizer
    regexTokenizer = RegexTokenizer(inputCol="message", outputCol="tokens", pattern="\\W+")
    stages += [regexTokenizer]
    
    
    

    cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
    stages += [cv]

# 3. Convert the labels to numerical values using binariser
    indexer = StringIndexer(inputCol="spam/ham", outputCol="label")
    stages += [indexer]
    
# 4. Vectorise features using vectorassembler
    vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
    stages += [vecAssembler]
    

    [print('\n', stage) for stage in stages]


    pipeline = Pipeline(stages=stages)
    data = pipeline.fit(df).transform(df)
    print(data)
    train, test = data.randomSplit([0.7, 0.3], seed = 2018)
     
    #naiveBayesClassifierGo(data) uncomment when needed
    #SgdClassifierGoTest(train)
    pareg(train)
