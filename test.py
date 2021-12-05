from pyspark.ml.feature import *
from pyspark.ml import Pipeline
#import * from sklearn
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import nltk
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer

# Use English stemmer.
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.types import *
import json
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import *

from sklearn.metrics import accuracy_score

def prediction_vals(test):
    # with open('NBmodel' , 'rb') as f:
    #     clf_test_model=pickle.load(f)
        
    #     X=test.select('message').collect()
    
    #     X=[i['message'] for i in X]
    #     vectorizer = HashingVectorizer(n_features=100)
    
    #     X = vectorizer.fit_transform(X)
    #     X=X.toarray()
        
    #     y=test.select('label').collect()
    #     y=np.array([i[0] for i in np.array(y)])
        
    #     pred_vals_holder = clf_test_model.predict(X)
        
    #     acc=accuracy_score(y, pred_vals_holder)
    #     #accuracy print
    #     #acc=clf_test_model.score(X,y)
    #     print(acc)


        # with open('PAREGmodel' , 'rb') as f:
        #     clf2_test_model=pickle.load(f)
            
        #     X=test.select('message').collect()
        
        #     X=[i['message'] for i in X]
        #     vectorizer = HashingVectorizer(n_features=100)
        
        #     X = vectorizer.fit_transform(X)
        #     X=X.toarray()
            
        #     y=test.select('label').collect()
        #     y=np.array([i[0] for i in np.array(y)])
            
        #     pred_vals_holder = clf2_test_model.predict(X)
            
        #     acc=accuracy_score(y, pred_vals_holder)
        #     #accuracy print
        #     #acc=clf_test_model.score(X,y)
        #     print(acc)


        with open('SGDmodel' , 'rb') as f:
            clf2_test_model=pickle.load(f)
            
            X=test.select('message').collect()
        
            X=[i['message'] for i in X]
            vectorizer = HashingVectorizer(n_features=100)
        
            X = vectorizer.fit_transform(X)
            X=X.toarray()
            
            y=test.select('label').collect()
            y=np.array([i[0] for i in np.array(y)])
            
            pred_vals_holder = clf2_test_model.predict(X)
            
            acc=accuracy_score(y, pred_vals_holder)
            #accuracy print
            #acc=clf_test_model.score(X,y)
            print(acc)

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

    


def sender(df):
    
    stages = []

# 3. Convert the labels to numerical values using binariser
    indexer = StringIndexer(inputCol="spam/ham", outputCol="label")
    stages += [indexer]


    pipeline = Pipeline(stages=stages)
    data = pipeline.fit(df).transform(df)
    #print(data)
    #test = data.randomSplit([0.7, 0.3], seed = 2018)
     
    prediction_vals(data)
        
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
    df = spark.createDataFrame((Row(**d) for d in val_holder), schema)
    stop_words = stopwords.words('english')
    trial=udf(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df=df.withColumn('message',trial('message'))
    trial2=udf(lambda x: [stemmer.stem(y) for y in x])
    df=df.withColumn('message',trial2('message'))
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

