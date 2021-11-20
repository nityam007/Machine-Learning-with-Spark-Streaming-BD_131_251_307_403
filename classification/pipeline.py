from pyspark.ml.feature import *
from pyspark.ml import Pipeline

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def naiveBayesClassifierGo(train,test):
    
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train) # Fit the model
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