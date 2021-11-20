import pickle
import json
import importlib

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.types import *

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import BinaryClassificationEvaluator

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

# Process each stream - needs to run ML models
def process(rdd):
	
	global schema, spark
	
	# Collect all records
	records = rdd.collect()
	
	# List of dicts
	dicts = [i for j in records for i in list(json.loads(j).values())]
	
	if len(dicts) == 0:
		return
	
	# Create a DataFrame with each stream	
	df = spark.createDataFrame((Row(**d) for d in dicts), schema)
	df.select('message').show()

	#curr_pipeline = custom_model_pipeline(df)
	
	#curr_model = get_model()
	#curr_model.partial_fit(df.select('tweet'), df.select('sentiment'), classes=[0, 4])
	
	# Save the model to a file
	#pipeline.write().overwrite().save("./pipeline")
	#with open('model.pkl', 'wb') as f:
	#	pickle.dump(model, f)


# Main entry point for all streaming functionality
if __name__ == '__main__':

	# Create a DStream - represents the stream of data received from TCP source/data server
	# Each record in 'lines' is a line of text
	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	# TODO: check if split is necessary
	json_str = lines.flatMap(lambda x: x.split('\n'))
	
	# Process each RDD
	lines.foreachRDD(process)

	# Start processing after all the transformations have been setup
	ssc.start()             # Start the computation
	ssc.awaitTermination()  # Wait for the computation to terminate
