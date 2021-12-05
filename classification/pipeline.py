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

from sparknlp.base import Finisher, DocumentAssembler
from sparknlp.annotator import (Tokenizer, Normalizer,LemmatizerModel, StopWordsCleaner)

from pyspark.ml import Pipeline

'''from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs'''

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

counter_nb=0
clf=GaussianNB()

clf_sgd=linear_model.SGDClassifier()
counter_sgd=0

'''mbk = MiniBatchKMeans(init ='k-means++', n_clusters = 4,
                      batch_size = batch_size, n_init = 10,
                      max_no_improvement = 10, verbose = 0)'''

mbk_counter=0

def ClusteringGo(train):
    # Load data in X 
    batch_size = 100
    global mbk
    global mbk_counter
    
    mbk_counter+=1
    
    X=train.select('message').collect()
    
    X=[i['message'] for i in X]
    
    ps = PorterStemmer()
    
    for wd in X:
        wd=ps.stem(wd)
        
    stop_words=['a','an','the','.',',',' \n','\n','i','me','my','myself','we','our','ours','ourselves','theirs','themselves','what','which','who','whom','this','that','these','those']

    vectorizer = HashingVectorizer(n_features=100000,stop_words=stop_words)
    
   
    
    X = vectorizer.fit_transform(X)
    
   
   
    y=train.select('label').collect()
    y=np.array([i[0] for i in np.array(y)])
    
    centers = [[1, 1], [-2, -1], [1, -2], [1, 9]]
    
    n_clusters = len(centers)
    
    X, labels_true = make_blobs(n_samples = 3000,centers = centers,cluster_std = 0.9)
  
    
    
  
    mbk.partial_fit(X)
    
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis = 0)
    
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
  
    # print the labels of each data
    
    print(mbk_means_labels)


def SgdClassifierGoTest(train):
    global clf_sgd
    global counter_sgd
    counter_sgd+=1
    
    X=train.select('message').collect()
    
    X=[i['message'] for i in X]
    
    stop_words=['a','an','the','.',',',' \n','\n','i','me','my','myself','we','our','ours','ourselves','theirs','themselves','what','which','who','whom','this','that','these','those']

    vectorizer = HashingVectorizer(n_features=100000,stop_words=stop_words)
    
    ps = PorterStemmer()
    
    for wd in range(len(X)):
        X[wd]=ps.stem(X[wd])
   
    
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
    
    #nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    
    global clf
    global counter_nb
    
    counter_nb+=1
    
    X=train.select('message').collect()
    
    X=[i['message'] for i in X]
    stop_words=['a','an','the','.',',',' \n','\n','i','me','my','myself','we','our','ours','ourselves','theirs','themselves','what','which','who','whom','this','that','these','those']

    vectorizer = HashingVectorizer(n_features=100000,stop_words=stop_words)
    
    ps = PorterStemmer()
    
    for wd in range(len(X)):
        X[wd]=ps.stem(X[wd])
    
    X = vectorizer.fit_transform(X)
   
   
    y=train.select('label').collect()
    y=np.array([i[0] for i in np.array(y)])
    
   
    
    clf.partial_fit(X.toarray(), y, classes = np.unique(y))
    
    if counter_nb ==30: 
        #save model into pickle file 
        with open('NBmodelWithStemmingFor500_2ndtime', 'wb') as files:
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
    
    #remover = StopWordsRemover(inputCol='tokens', outputCol='words_clean')
    #df_words_no_stopw = remover.transform(df).select('message', 'words_clean')
    
    #val_remover =StopWordsRemover(inputCol='tokens', outputCol='words_clean')

    '''# Stem text
    stemmer = SnowballStemmer(language='english')
    stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in words_clean], ArrayType(StringType()))
    df_stemmed = df_words_no_stopw.withColumn("words_stemmed", stemmer_udf("words_clean")).select('message', 'words_stemmed')

    lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['message']) \
     .setOutputCol('lemma')
    stages+=[lemmatizer]'''
    
# 2. CountVectorize the data
    '''vectorizer = HashingVectorizer(n_features=2**4)
    
    X=df.select('tokens')
    
    print('After selecting X:',X)
 
    
    X = vectorizer.fit_transform(corpus)
    '''
    cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
    stages += [cv]

# 3. Convert the labels to numerical values using binariser
    indexer = StringIndexer(inputCol="spam/ham", outputCol="label")
    stages += [indexer]
    
# 4. Vectorise features using vectorassembler
    vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
    stages += [vecAssembler]
    
    '''stemmer = SnowballStemmer(language='english')
    stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in words_clean], ArrayType(StringType()))
    stages+=[stemmer]'''
    
    [print('\n', stage) for stage in stages]


    pipeline = Pipeline(stages=stages)
    data = pipeline.fit(df).transform(df)
    print(data)
    #train, test = data.randomSplit([0.7, 0.3], seed = 2018)
     
    naiveBayesClassifierGo(data)
    #SgdClassifierGoTest(data)     #when to call sgd classifier 
    #ClusteringGo(data)


'''def sender(df):
    documentAssembler = DocumentAssembler() \
     .setInputCol('message') \
     .setOutputCol('document')
     
    tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('token')# note normalizer defaults to changing all words to lowercase.
     
# Use .setLowercase(False) to maintain input case.
    normalizer = Normalizer() \
     .setInputCols(['token']) \
     .setOutputCol('normalized') \
     .setLowercase(True)# note that lemmatizer needs a dictionary. So I used the pre-trained
     
# model (note that it defaults to english)
    lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemma')
     
    stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemma']) \
     .setOutputCol('clean_lemma') \
     .setCaseSensitive(False) \
     .setStopWords(eng_stopwords)# finisher converts tokens to human-readable output
    
    finisher = Finisher() \
     .setInputCols(['clean_lemma']) \
     .setCleanAnnotations(False)
    
    indexer = StringIndexer(inputCol="spam/ham", outputCol="label")
    
    pipeline = Pipeline() \
     .setStages([
           documentAssembler,
           tokenizer,
           normalizer,
           lemmatizer,
           stopwords_cleaner,
           finisher,
           indexer
     ])   

    df = pipeline.fit(df).transform(df)
    
    print(df)
    '''
