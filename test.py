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
from sklearn.decomposition import PCA
import json
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.types import *

from pyspark.sql.functions import *
from pyspark.sql.types import *


from pyspark.ml.feature import *

from classification import pipeline

from sklearn.metrics import accuracy_score

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report


def clustering_pred_vals(test):
    with open('MiniBatchKmeans_500' , 'rb') as f:
        clf_test_model=pickle.load(f)
        
        X=test.select('message').collect()
    
        X=[i['message'] for i in X]
        
        
        stop_words=['a','an','the','.',',',' \n','\n','i','me','my','myself','we','our','ours','ourselves','theirs','themselves','what','which','who','whom','this','that','these','those']
    
        vectorizer = HashingVectorizer(n_features=5000,stop_words=stop_words)
        
        
    	
        ps = PorterStemmer()
    
        for wd in range(len(X)):
            X[wd]=ps.stem(X[wd])
        
        X = vectorizer.fit_transform(X)
        X=X.toarray()
        #print(X[100])
        #print(X.shape)
        y=test.select('label').collect()
        y=np.array([i[0] for i in np.array(y)])
        
        pred_vals_holder = clf_test_model.predict(X)
        #X=X.reshape(-1)
        
        print(np.mean(pred_vals_holder==y))
        
        print(classification_report(y,pred_vals_holder))
        
        #print( len(pred_vals_holder))
        
        '''pca = PCA(n_components= 2,svd_solver='arpack')
        scatter_plot_points = pca.fit_transform(X)

        colors = ["r", "b"]

        x_axis = [o[0] for o in scatter_plot_points]
        y_axis = [o[1] for o in scatter_plot_points]
        plt.scatter(x_axis,y_axis,c=[colors[d] for d in pred_vals_holder])
        #Plotting the results
        
        plt.savefig('clusters.png')'''
        
        
        #acc=accuracy_score(y, pred_vals_holder)
        #accuracy print
        #acc=clf_test_model.score(X,y)
       # print(np.mean(pred_vals_holder==y))
       # print(classification_report(y,pred_vals_holder))

def prediction_vals(test):
    with open('NBmodelWithStemmingFor500' , 'rb') as f:
        clf_test_model=pickle.load(f)
        
        X=test.select('message').collect()
    
        X=[i['message'] for i in X]
        
        
        stop_words=['a','an','the','.',',',' \n','\n','i','me','my','myself','we','our','ours','ourselves','theirs','themselves','what','which','who','whom','this','that','these','those']
    
        vectorizer = HashingVectorizer(n_features=100000)
        
        
    	
        ps = PorterStemmer()
    
        for wd in range(len(X)):
            X[wd]=ps.stem(X[wd])
        
        X = vectorizer.fit_transform(X)
        X=X.toarray()
        
        y=test.select('label').collect()
        y=np.array([i[0] for i in np.array(y)])
        
        pred_vals_holder = clf_test_model.predict(X)
        
        acc=accuracy_score(y, pred_vals_holder)
        #accuracy print
        #acc=clf_test_model.score(X,y)
        print(np.mean(pred_vals_holder==y))
        print(classification_report(y,pred_vals_holder))

sc = SparkContext("local[2]", "spam")
    
spark = SparkSession \
.builder \
.config(conf=SparkConf()) \
.getOrCreate()

# Batch interval of 5 seconds - TODO: Change according to necessity
ssc = StreamingContext(sc, 5)

sql_context = SQLContext(sc)
    
# Set constant for the TCP port to send from/listen to
TCP_IP = "localhost"
TCP_PORT = 6100
    
# Create schema
schema = StructType([
    StructField("subject", StringType(), False),
    StructField("message", StringType(), False),
    StructField("spam/ham", StringType(), False),
])

    
# def cleaner(df):
#     tokenizer1 = Tokenizer(inputCol="subject", outputCol="revised_subject")
#     tokenizer2 = Tokenizer(inputCol="message", outputCol="revised_message")
#     df = tokenizer1.transform(df)	 #Transforms the dataset into a new dataset after applying all the changes
#     df = tokenizer2.transform(df)	 #Transforms the dataset into a new dataset after applying all the changes
#                                 # Needs no saving
#     remover = StopWordsRemover(inputCol="revised_subject", outputCol="filtered_subject")
    
#     df = remover.transform(df).show(truncate=False)
#     #df.show()

def sender(df):
    stages = []
# 1. clean data and tokenize sentences using RegexTokenizer
    regexTokenizer = RegexTokenizer(inputCol="message", outputCol="tokens", pattern="\\W+")
    stages += [regexTokenizer]

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
    
    #[print('\n', stage) for stage in stages]


    pipeline = Pipeline(stages=stages)
    data = pipeline.fit(df).transform(df)
    #print(data)
    #test = data.randomSplit([0.7, 0.3], seed = 2018)
     
    prediction_vals(data)
    
    #clustering_pred_vals(data)
    
    #randomForestCLassifier(train,test,data)
        
# Process each stream - needs to run ML models
def process(rdd):
    
    global schema, spark
    
    # Collect all records
    rdds = rdd.collect()
    
    # List of dicts
    val_holder = [i for j in rdds for i in list(json.loads(j).values())]
    
    if len(val_holder) == 0:
        return
    
    # Create a DataFrame with each stream	
    df = spark.createDataFrame((Row(**d) for d in val_holder), schema)
    
    #df.show()
    
    #cleaner(df)
    sender(df)

    

if __name__ == '__main__':

    # Create a DStream - represents the stream of data received from TCP source/data server
    # Each record in 'lines' is a line of text
    read_lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

    # TODO: check if split is necessary
    line_vals = read_lines.flatMap(lambda x: x.split('\n'))
    
    # Process each RDD
    read_lines.foreachRDD(process)

    # Start processing after all the transformations have been setup
    ssc.start()             # Start the computation
    ssc.awaitTermination()  # Wait for the computation to terminate
