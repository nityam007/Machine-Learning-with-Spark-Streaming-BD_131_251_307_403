from pyspark.ml.feature import *
from pyspark.ml import Pipeline
#import * from sklearn
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
from sklearn import linear_model

#from sparknlp.base import Finisher, DocumentAssembler
#from sparknlp.annotator import (Tokenizer, Normalizer,
                         #LemmatizerModel, StopWordsCleaner)
from pyspark.ml import Pipeline


counter_nb=0
clf=GaussianNB()

clf_sgd=linear_model.SGDClassifier()
counter_sgd=0

def SgdClassifierGoTest(train):
    global clf_sgd
    global counter_sgd
    counter_sgd+=1
    
    X=train.select('message').collect()
    
    X=[i['message'] for i in X]
    
    stop_words=['a','an','the','.',',',' \n','\n','i','me','my','myself','we','our','ours','ourselves','theirs','themselves','what','which','who','whom','this','that','these','those']

    vectorizer = HashingVectorizer(n_features=100000,stop_words=stop_words)
    
   
    
    X = vectorizer.fit_transform(X)
    
   
   
    y=train.select('label').collect()
    y=np.array([i[0] for i in np.array(y)])
    
   
    
    clf_sgd.partial_fit(X.toarray(), y, classes = np.unique(y))
  
    if counter_sgd == 30: #counter_sgd is 30 when batch size is 1000 and 303 when batch size is 100
        #save model into pickle file 
        
        with open('SGDmodel_1000', 'wb') as files:
             
             pickle.dump(clf_sgd, files)
        return
    
    linear_model.SGDClassifier()


def naiveBayesClassifierGo(train):
    
    #nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    
    global clf
    global counter_nb
    
    counter_nb+=1
    
    X=train.select('message').collect()
    
    X=[i['message'] for i in X]
    stop_words=['a','an','the','.',',',' \n','\n','i','me','my','myself','we','our','ours','ourselves','theirs','themselves','what','which','who','whom','this','that','these','those']

    vectorizer = HashingVectorizer(n_features=100000,stop_words=stop_words)
    
   
    
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
    
    
    '''predictions = clf.transform(test)
    predictions.select("label", "prediction", "probability").show()
    
    
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print ("Model Accuracy: ", accuracy)'''


def sender(df):
    stages = []
    
    # 1. clean data and tokenize sentences using RegexTokenizer
    regexTokenizer = RegexTokenizer(inputCol="message", outputCol="tokens", pattern="\\W+")
    stages += [regexTokenizer]
    
    '''remover = StopWordsRemover(inputCol='tokens', outputCol='words_clean')
    df_words_no_stopw = remover.transform(df).select('message', 'words_clean')'''
    
    #val_remover =StopWordsRemover(inputCol='tokens', outputCol='words_clean')

    # Stem text
    '''stemmer = SnowballStemmer(language='english')
    stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in words_clean], ArrayType(StringType()))
    df_stemmed = df_words_no_stopw.withColumn("words_stemmed", stemmer_udf("words_clean")).select('message', 'words_stemmed')'''

    '''lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['message']) \
     .setOutputCol('lemma')'''
    
# 2. CountVectorize the data
    '''vectorizer = HashingVectorizer(n_features=2**4)
    
    X=df.select('tokens')
    
    print('After selecting X:',X)
 
    
    X = vectorizer.fit_transform(corpus)'''
    
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
    #train, test = data.randomSplit([0.7, 0.3], seed = 2018)
     
    #naiveBayesClassifierGo(data)
    SgdClassifierGoTest(data)     #when to call sgd classifier 
  
    
    
    
    
