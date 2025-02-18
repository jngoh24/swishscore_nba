# Databricks notebook source
# MAGIC %md
# MAGIC Import Necessary Functions

# COMMAND ----------

from pyspark.sql.functions import col, count, upper, concat, lit, when, split, size, concat_ws
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC Load and Clean Data

# COMMAND ----------

teams = spark.read.csv("dbfs:/mnt/cinqai_outputs/jn_adhoc/lead_scoring/data/nba_data/team_data/21.22_teamdata.csv", header=True, inferSchema=True)

teams = teams.withColumn("TEAM_ABBRV", upper(teams["TEAM_ABBRV"]))

teams_abbrv = teams.select("TEAM", "TEAM_ABBRV")

teams_join = teams.select("TEAM_ABBRV", "oPPG", "dEFF")

teams_join = teams_join.withColumnRenamed("TEAM_ABBRV", "TEAM_DEF")

# COMMAND ----------

players_original = spark.read.csv("dbfs:/mnt/cinqai_outputs/jn_adhoc/lead_scoring/data/nba_data/player_data/21.22_playerdata.csv", header=True, inferSchema=True)

players = players_original.withColumn("Team", upper(players_original["TEAM"]))

players = players.dropna()

players_filtered = players.filter(players["GP"] >= 50)

players_new = players.withColumn("GP_Filter", when(players.GP >= 50, "Yes").otherwise("No"))

players_norm = players_new.withColumn("ppg_norm", col("ppg") / col("MPG"))

players_norm = players_norm.withColumn("key", concat(col("FULL NAME"), lit("_"), col("Team")))

players_join = players_norm.select("FULL NAME", "TEAM", "POS", "GP", "MPG", "PPG", "TS%", "eFG%", "2P%", "3P%", "ORTG", "GP_Filter", "key")

# COMMAND ----------

shots = spark.read.csv("dbfs:/mnt/cinqai_outputs/jn_adhoc/lead_scoring/data/nba_data/game_data/21.22_shotsdata.csv", header=True, inferSchema=True)

# Split the column by space
split_col = split(col("TEAM_NAME"), " ")

# Determine where to split based on word count
shots_split = shots.withColumn(
    "TEAM_NEW",
    when(size(split_col) == 3, concat_ws(" ", split_col[0], split_col[1]))  # First two words if three exist
    .otherwise(split_col[0])  # First word otherwise
).withColumn(
    "TEAM_NEW2",
    when(size(split_col) == 3, split_col[2])  # Last word if three exist
    .otherwise(split_col[1])  # Second word otherwise
)

shots_renamed = shots_split.withColumn(
    "TEAM",
    when((col("TEAM_NEW") == "Los Angeles") & (col("TEAM_NEW2") == "Lakers"), "LA Lakers")
    .when((col("TEAM_NEW") == "Los Angeles") & (col("TEAM_NEW2") == "Clippers"), "LA Clippers")
    .when((col("TEAM_NEW") == "Portland Trail") & (col("TEAM_NEW2") == "Blazers"), "Portland")
    .otherwise(col("TEAM_NEW"))
)

shots_renamed = shots_renamed.withColumn(
    "AWAY_TEAM",
    when(col("AWAY_TEAM") == "BKN", "BRO").otherwise(col("AWAY_TEAM"))
)
shots_with_abbrv = shots_renamed.join(teams_abbrv, on="TEAM", how="left")

shot_defend = shots_with_abbrv.withColumn(
    "TEAM_DEF",
    when(col("TEAM_ABBRV") == col("HOME_TEAM"), col("AWAY_TEAM")).otherwise(col("HOME_TEAM"))
)

shot_cat = shot_defend.withColumn("SHOT_CAT", when(col("SHOT_MADE") == "true", 1).otherwise(0))

shots_final = shot_cat.withColumn("key", concat(col("PLAYER_NAME"), lit("_"), col("TEAM_ABBRV")))

# COMMAND ----------

master_df1 = shots_final.join(players_join, on="key", how="left")
final_master = master_df1.join(teams_join, on="TEAM_DEF", how="left")

# COMMAND ----------

final_master_filter = final_master.filter(final_master.GP_Filter == "Yes")

# COMMAND ----------

final_master.count()

# COMMAND ----------

final_master_filter.count()

# COMMAND ----------

display(final_master.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC EDA

# COMMAND ----------

print(f"There are {players.count()} players that played in the 2021-22 season. Of those, {players_filtered.count()} played 50+ games that season.")

# COMMAND ----------

# MAGIC %md
# MAGIC EDA - Shots

# COMMAND ----------

from pyspark.sql.functions import col, count, sum, round

shots_made = final_master_filter.groupBy("event_type") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc())

# Compute total count of shots
total_shots = shots_made.agg(sum("count").alias("total")).collect()[0]["total"]

# Add percentage of total column
shots_made = shots_made.withColumn("percentage", round((col("count") / total_shots) * 100, 2))

display(shots_made)

# COMMAND ----------

action_types = shots.groupBy("action_type") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc()) \

display(action_types.limit(10))

# COMMAND ----------

zones = shots.groupBy("ZONE_NAME") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc()) \

display(zones)

# COMMAND ----------

basic_zones = shots.groupBy("BASIC_ZONE") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc()) \

display(basic_zones)

# COMMAND ----------

shot_range = shots.groupBy("ZONE_RANGE") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc()) \

display(shot_range)

# COMMAND ----------

shots_per_quarter = shots.groupBy("QUARTER") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc()) \

display(shots_per_quarter)

# COMMAND ----------

# MAGIC %md
# MAGIC EDA - Teams

# COMMAND ----------

top_10_shot_teams = shots.groupBy("TEAM_NAME") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc()) \

display(top_10_shot_teams.limit(10))

# COMMAND ----------

bottom_10_shot_teams = shots.groupBy("TEAM_NAME") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").asc()) \
  
display(bottom_10_shot_teams.limit(10))

# COMMAND ----------

oPPg = teams.select("TEAM", "oPPG")
top10_oPPg = oPPg.orderBy(col("oPPG").asc()).limit(10)
display(top10_oPPg)

# COMMAND ----------

dEFF = teams.select("TEAM", "dEFF")
top10_dEFF = dEFF.orderBy(col("dEFF").asc()).limit(10)
display(top10_dEFF)

# COMMAND ----------

bottom10_oPPg = oPPg.orderBy(col("oPPG").desc()).limit(10)
display(bottom10_oPPg)

# COMMAND ----------

bottom10_dEFF = dEFF.orderBy(col("dEFF").desc()).limit(10)
display(bottom10_dEFF)

# COMMAND ----------

# MAGIC %md
# MAGIC EDA - Players

# COMMAND ----------

top_10_shot_takers = final_master_filter.groupBy("player_name") \
  .agg(count("*").alias("count")) \
  .orderBy(col("count").desc()) \

display(top_10_shot_takers.limit(10))

# COMMAND ----------

eFG = players_filtered.select("FULL NAME", "eFG%")
top10_eFG = eFG.orderBy(col("eFG%").desc()).limit(10)
display(top10_eFG)

# COMMAND ----------

TS = players_filtered.select("FULL NAME", "TS%")
top10_TS = TS.orderBy(col("TS%").desc()).limit(10)
display(top10_TS)

# COMMAND ----------

two_pt = players_filtered.select("FULL NAME", "2P%")
top10_two_pt = two_pt.orderBy(col("2P%").desc()).limit(10)
display(top10_two_pt)

# COMMAND ----------

three_pt = players_filtered.select("FULL NAME", "3P%")
top10_three_pt = three_pt.orderBy(col("3P%").desc()).limit(10)
display(top10_three_pt)

# COMMAND ----------

ORTG = players_filtered.select("FULL NAME", "ORTG")
top10_ORTG = ORTG.orderBy(col("ORTG").desc()).limit(10)
display(top10_ORTG)

# COMMAND ----------

# MAGIC %md
# MAGIC Correlations

# COMMAND ----------

numerical_cols = ['GP', 'MPG', 'PPG', 'TS%', 'eFG%', '2P%', '3P%', 'ORTG', 'oPPG', 'dEFF']

for col in numerical_cols:
    correlation = final_master_filter.stat.corr("SHOT_CAT", col)
    print(f"Correlation between binary_target and {col}: {correlation}")

# COMMAND ----------

from dython.nominal import associations

master_pandas = final_master_filter.select("TEAM_ABBRV", "TEAM_DEF", "POSITION", "ACTION_TYPE", "SHOT_TYPE", "BASIC_ZONE",
                                           "ZONE_NAME", "ZONE_RANGE", "SHOT_DISTANCE", "QUARTER", "SHOT_CAT").toPandas()

corr_matrix = associations(master_pandas,  figsize=(10, 8))

# COMMAND ----------

corrs = corr_matrix['corr'].round(2)
shot_corrs = corrs["SHOT_CAT"]
high_shot_corrs = shot_corrs[shot_corrs > 0.07]
print(high_shot_corrs)

# COMMAND ----------

# MAGIC %md
# MAGIC Model Building

# COMMAND ----------

master_model = final_master_filter.select("FULL NAME", "TEAM_ABBRV", "POSITION", "ACTION_TYPE", "SHOT_TYPE", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE", "TS%", "eFG%", "2P%", "ORTG", "SHOT_CAT")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# COMMAND ----------

categorical_cols = ["POSITION", "ACTION_TYPE", "SHOT_TYPE", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE"]
numerical_cols = ["TS%", "eFG%", "2P%", "ORTG"]

# Indexing and One-Hot Encoding for categorical variables
indexers = [StringIndexer(inputCol=col, outputCol=col+"_indexed") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_indexed", outputCol=col+"_encoded") for col in categorical_cols]

# Assembling features
assembler = VectorAssembler(
    inputCols=numerical_cols + [col+"_encoded" for col in categorical_cols],
    outputCol="features"
)

# COMMAND ----------

train_data, test_data = master_model.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

lr = LogisticRegression(featuresCol="features", labelCol="SHOT_CAT")

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

model = pipeline.fit(train_data)

# COMMAND ----------

predictions = model.transform(test_data)

# predictions.select("SHOT_CAT", "probability", "prediction").show()

# COMMAND ----------

# Accuracy Evaluator
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="SHOT_CAT", predictionCol="prediction", metricName="accuracy"
)

# Calculate accuracy
accuracy = accuracy_evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.2f}")

# COMMAND ----------

prediction_and_labels = predictions.select("prediction", "SHOT_CAT").rdd.map(lambda x: (float(x[0]), float(x[1])))

# Compute confusion matrix
metrics = MulticlassMetrics(prediction_and_labels)
confusion_matrix = metrics.confusionMatrix().toArray()

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)

# COMMAND ----------

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="SHOT_CAT", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# Calculate AUC
auc = auc_evaluator.evaluate(predictions)
print(f"AUC: {auc:.2f}")

# COMMAND ----------

# Extract the Logistic Regression model
lr_model = model.stages[-1]

# Get coefficients and intercept
coefficients = lr_model.coefficients
intercept = lr_model.intercept

# List of feature names
feature_names = numerical_cols + [f"{col}_encoded" for col in categorical_cols]

# Display Feature Importance
print(f"Intercept: {intercept}")
print("Feature Importance (Coefficients):")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef}")

# COMMAND ----------

