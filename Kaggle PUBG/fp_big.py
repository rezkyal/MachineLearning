# Konfigurasi Spark
import os
import sys

# 1. Mengeset variabel yang menyimpan lokasi di mana Spark diinstal
spark_path = "D:\\spark-2.3.0-bin-hadoop2.7"

# 2. Menentukan environment variable SPARK_HOME
os.environ['SPARK_HOME'] = spark_path

# 3. Simpan lokasi winutils.exe sebagai environment variable HADOOP_HOME
os.environ['HADOOP_HOME'] = "D:\\Program Files\\hadoop\\hadoop-2.8.3"

# 4. Lokasi Python yang dijalankan --> punya Anaconda
#    Apabila Python yang diinstall hanya Anaconda, maka tidak perlu menjalankan baris ini.
os.environ['PYSPARK_PYTHON'] = sys.executable

# 5. Konfigurasi path library PySpark
sys.path.append(spark_path + "/bin")
sys.path.append(spark_path + "/python")
sys.path.append(spark_path + "/python/pyspark/")
sys.path.append(spark_path + "/python/lib")
sys.path.append(spark_path + "/python/lib/pyspark.zip")
sys.path.append(spark_path + "/python/lib/py4j-0.10.6-src.zip")

# 6. Import library Spark
#    Dua library yang WAJIB di-import adalah **SparkContext** dan **SparkConf**.
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession,functions as F
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, DecisionTreeRegressionModel
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import ChiSqSelector, VectorSlicer

# Setting konfigurasi (opsional)
conf = SparkConf()
conf.set("spark.executor.memory", "8g")
conf.set("spark.cores.max", "4")

sc = SparkContext("local", conf=conf)
#    Apabila berhasil, maka ketika sc di-print akan mengeluarkan nilai <pyspark.context.SparkContext object>
print(sc)

# 1. Data preparation
sql = SQLContext(sc)
parsedData = sql.read \
    .format("com.databricks.spark.csv") \
    .option("header", "true") \
    .load("agg_match_stats_4.csv")

spark = SparkSession \
    .builder \
    .config("spark.some.config.option","some-value") \
    .getOrCreate()

# df = spark.read.csv("agg_match_stats_4.csv").rdd

# Data selection and preprocessing

# Change the datatype of each attribute to how is supposed to be
parsedData=parsedData\
    .withColumn("date", parsedData["date"].cast("Timestamp")) \
    .withColumn("game_size", parsedData["game_size"].cast("integer")) \
    .withColumn("party_size", parsedData["party_size"].cast("double")) \
    .withColumn("team_id", parsedData["team_id"].cast("integer")) \
    .withColumn("team_placement", parsedData["team_placement"].cast("double")) \
    .withColumn("player_kills", parsedData["player_kills"].cast("integer")) \
    .withColumn("player_dbno", parsedData["player_dbno"].cast("integer")) \
    .withColumn("player_assists", parsedData["player_assists"].cast("integer")) \
    .withColumn("player_dmg", parsedData["player_dmg"].cast("integer")) \
    .withColumn("player_dist_ride", parsedData["player_dist_ride"].cast("double")) \
    .withColumn("player_dist_walk", parsedData["player_dist_walk"].cast("double"))
parsedData.show(5)

# Select the used attribute only using sparkSQL
parsedData.createOrReplaceTempView("data")
usedata= spark.sql("Select game_size, party_size,team_placement,player_kills,player_dbno,player_assists,player_dmg,player_dist_ride,player_dist_walk FROM data Where party_size=1")

# Change the target class from the ranking of the team (rank 1-100) to chicken dinner or not (1 for chicken dinner and 0 for not)
rank=(F.when(F.col('team_placement') == 1,1).otherwise(0))
usedatas = usedata.withColumn("rank",rank)
usedatas.show()

# Data transformation
# Using chi-square to pick 2 most influential attribute to the classification
assembler=VectorAssembler(
    inputCols=["game_size","player_kills","player_dbno","player_assists","player_dmg"],
    outputCol="selectedFeatures")
transformed = assembler.transform(usedatas)
selector = ChiSqSelector(numTopFeatures=2,featuresCol="selectedFeatures", outputCol="features", labelCol="team_placement")
transformed = selector.fit(transformed).transform(transformed)
transformed.show()

# Separate data into trainingSet and testSet with random split
trainingSet,testSet=transformed.randomSplit([0.7,0.3])




# Modelling & Evaluating
dt=DecisionTreeClassifier(maxDepth=10, labelCol="rank")
rf=RandomForestClassifier(maxDepth=10,labelCol="rank")
cv1=CrossValidator(estimator=dt,evaluator=evaluator,numFolds=3)
cv2=CrossValidator(estimator=rf,evaluator=evaluator,numFolds=3)
model1=dt.fit(trainingSet)
model2=rf.fit(trainingSet)
prediction1 = model1.transform(testSet)\
    .selectExpr("features","team_placement as label","prediction")
prediction2 = model1.transform(testSet)\
    .selectExpr("features","team_placement as label","prediction")

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
print("Menggunakan DecisionTree:")
evaluator.evaluate(prediction1)
print("Menggunakan RandomForest:")
evaluator.evaluate(prediction2)
