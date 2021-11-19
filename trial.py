from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import pandas as pd
import json 
from pyspark.sql import SparkSession
import sys
import re
import operator
from pyspark.ml import Pipeline
import numpy 


# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

spark = SparkSession.builder.appName('sparkdf').getOrCreate()

lines = ssc.socketTextStream("localhost", 6100)

#json_line = json.loads(lines)
#words = json_lines.flatMap(lambda json_line: json_line.split("\n"))

def readmystream(rdd):
	df = spark.read.json(rdd)

	#df.printschema()
	
	#df.show()	
	
	f=df.collect()
	#df.show()
	 
	f0=[]
	f1=[]
	f2=[]
	
	for i in f:
		for k in i:
			#for l in k:
			f0.append(k[0])
			f1.append(k[1])
			f2.append(k[2])
	print(f0)
	print(len(f0))
	
	print(f1)
	print(len(f1))
	
	print(f2)
	print(len(f2))
	
		
		#print(f"{i}\t{j}\t{k}")
	
	

lines.foreachRDD(lambda rdd: readmystream (rdd))


ssc.start()

ssc.awaitTermination()

ssc.stop()
