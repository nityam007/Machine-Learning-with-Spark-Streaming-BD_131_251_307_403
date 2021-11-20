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
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.classification import NaiveBayes # Initialise the model

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

def naiveBayesClassifierGo(train,test):
	
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")# Fit the model
	model = nb.fit(train)# Make predictions on test data
	predictions = model.transform(test)
	predictions.select("label", "prediction", "probability").show()
	
	
	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Model Accuracy: ", accuracy)
	
def sender(df):
	stages = []
# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="subject", outputCol="tokens", pattern="\\W+")
	stages += [regexTokenizer]

# 2. CountVectorize the data
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
	train, test = data.randomSplit([0.7, 0.3], seed = 2018)
	 
	naiveBayesClassifierGo(train,test)

	
def cleaner(df):
	tokenizer1 = Tokenizer(inputCol="subject", outputCol="revised_subject")
	tokenizer2 = Tokenizer(inputCol="message", outputCol="revised_message")
	df = tokenizer1.transform(df)	 #Transforms the dataset into a new dataset after applying all the changes
	df = tokenizer2.transform(df)	 #Transforms the dataset into a new dataset after applying all the changes
								# Needs no saving
	remover = StopWordsRemover(inputCol="revised_subject", outputCol="filtered_subject")
	
	df = remover.transform(df).show(truncate=False)
	#df.show()
	
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
	read_lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	# TODO: check if split is necessary
	line_vals = read_lines.flatMap(lambda x: x.split('\n'))
	
	# Process each RDD
	read_lines.foreachRDD(process)

	# Start processing after all the transformations have been setup
	ssc.start()             # Start the computation
	ssc.awaitTermination()  # Wait for the computation to terminate
