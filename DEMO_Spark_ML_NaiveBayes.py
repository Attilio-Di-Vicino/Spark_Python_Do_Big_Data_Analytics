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

#Create a Spark Session
SpSession = SparkSession.builder \
            .appName("a_naive_bayes_classifier_for_spam_sms") \
            .getOrCreate()

#TEST
# SpSession = SparkSession.builder.master("local-cluster[4,1,1]") \
#             .appName("a_naive_bayes_classifier_for_spam_sms") \
#             .getOrCreate()

#Initialize the Spark context
SpContext = SpSession.sparkContext

"""--------------------------------------------------------------------------
Print some info
-------------------------------------------------------------------------"""
print("Application ID:", SpContext.applicationId)
print("Master:", SpContext.getConf().get("spark.master"))
print("Deploy Mode:", SpContext.getConf().get("spark.submit.deployMode"))
print("Number of Executors:", SpContext.getConf().get("spark.executor.instances", "N/A"))
print("Executor Memory:", SpContext.getConf().get("spark.executor.memory", "N/A"))
print("Executor Cores:", SpContext.getConf().get("spark.executor.cores", "N/A"))
print("Driver Memory:", SpContext.getConf().get("spark.driver.memory", "N/A"))


"""--------------------------------------------------------------------------
Load Data
-------------------------------------------------------------------------"""
#Load the CSV file into a RDD
file_name = "SMSSpamCollection.csv"
gcs_path = "gs://dataproc-staging-us-central1-318082001328-fmcfsyci/SMSSpamCollection.csv"
#Read a text file from HDFS, a local file system (available on all nodes),
#or any Hadoop-supported file system URI, and return it as an RDD of Strings.
#The text files must be encoded as UTF-8.
smsData = SpContext.textFile(gcs_path,2)
smsData.cache()
#Returns all the records as a list of Row.
smsData.collect()

"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""
#Convert into DataFrame start from RDD
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
print(f"Accuracy: {accuracy * 100:.2f}%\n\n")

#Stop Spark Contex
SpContext.stop()