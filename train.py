import json

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.types import *

from pyspark.sql.functions import *
from pyspark.sql.types import *


from pyspark.ml.feature import *

from classification import pipeline
#from classification_models.pipeline_sparkml import custom_model_pipeline, ml_algorithm

# Create a local StreamingContext with two execution threads
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
    pipeline(df)

    

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
