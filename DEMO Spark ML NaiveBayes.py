# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------

           Naive Bayes : Spam Filtering

             Copyright : V2 Maestros @2016

Problem Statement
*****************
The input data is a set of SMS messages that has been classified
as either "ham" or "spam". The goal of the exercise is to build a
 model to identify messages as either ham or spam.

## Techniques Used

1. Naive Bayes Classifier
2. Training and Testing
3. Confusion Matrix
4. Text Pre-Processing
5. Pipelines

-----------------------------------------------------------------------------
"""
#import os
#os.chdir("C:/Personal/V2Maestros/Courses/Big Data Analytics with Spark/Python")
#os.curdir

"""--------------------------------------------------------------------------
Init
-------------------------------------------------------------------------"""
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Create a Spark Session
SpSession = SparkSession.builder.master("local[2]") \
            .appName("Attilio-Di-Vicino") \
            .getOrCreate()

# Initialize the Spark context
SpContext = SpSession.sparkContext


"""--------------------------------------------------------------------------
Load Data
-------------------------------------------------------------------------"""
#Load the CSV file into a RDD
smsData = SpContext.textFile("SMSSpamCollection.csv",2)
smsData.cache()
smsData.collect()

"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""

def TransformToVector(inputStr):
    attList=inputStr.split(",")
    smsType= 0.0 if attList[0] == "ham" else 1.0
    return [smsType, attList[1]]

smsXformed=smsData.map(TransformToVector)

smsDf= SpSession.createDataFrame(smsXformed,
                          ["label","message"])
smsDf.cache()
smsDf.select("label","message").show()

"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""
#Split training and testing
(trainingData, testData) = smsDf.randomSplit([0.9, 0.1])
trainingData.count()
testData.count()
testData.collect()

#Setup pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Split into words and then build TF-IDF
tokenizer = Tokenizer(inputCol="message", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), \
        outputCol="tempfeatures")
idf=IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
nbClassifier=NaiveBayes()

pipeline = Pipeline(stages=[tokenizer, hashingTF, \
                idf, nbClassifier])

#Build a model with a pipeline
nbModel=pipeline.fit(trainingData)
#Predict on test data (will automatically go through pipeline)
prediction=nbModel.transform(testData)

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
accuracy = evaluator.evaluate(prediction)

#Draw confusion matrics
prediction.groupBy("label","prediction").count().show()

#Print accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

#Stop Spark Contex
SpContext.stop()