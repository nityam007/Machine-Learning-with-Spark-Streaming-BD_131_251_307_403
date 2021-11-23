from pyspark.ml.feature import *
from pyspark.ml import Pipeline

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
def naiveBayesClassifierGo(train,test):
    
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train) # Fit the model
    predictions = model.transform(test)
    predictions.select("label", "prediction", "probability").show()
    
    
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print ("Model Accuracy: ", accuracy)



def randomForestCLassifier(train,test,data):

    labelIndexer = StringIndexer(inputCol="spam/ham", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    #(trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select example rows to display.
    predictions.select("spam/ham","predictedLabel").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))



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
     
    #naiveBayesClassifierGo(train,test)
    randomForestCLassifier(train,test,data)
