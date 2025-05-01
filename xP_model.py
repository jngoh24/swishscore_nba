# Databricks notebook source
# MAGIC %md
# MAGIC ## **Import Necessary Functions**

# COMMAND ----------

from pyspark.sql.functions import col, count, upper, concat, lit, when, split, size, concat_ws, lag, monotonically_increasing_id, element_at, round
from pyspark.sql.functions import sum as _sum
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Load and Clean Data**

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

final_master = final_master.orderBy(
    col("game_id").asc(),
    col("quarter").asc(),
    col("mins_left").desc(),
    col("secs_left").desc()
)

# COMMAND ----------

final_master = final_master.withColumn("shot_order", monotonically_increasing_id())

# COMMAND ----------

# Define a window partitioned by game_id and player_id, maintaining shot log order
window_spec = Window.partitionBy("game_id").orderBy(
    col("quarter").asc(), col("mins_left").desc(), col("secs_left").desc()
).rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Calculate cumulative shots taken per player per game
final_master_new = final_master.withColumn("shots_taken", count("player_id").over(window_spec) - 1)

# Calculate cumulative shots made per player per game (only when shot_made = true)
final_master_new = final_master_new.withColumn(
    "shots_made",
    F.sum(when(col("shot_made") == True, 1).otherwise(0)).over(window_spec) - 1
)

# Ensure the first shot in each game for a player starts with 0
first_shot_window = Window.partitionBy("game_id", "player_id").orderBy(
    col("quarter").asc(), col("mins_left").desc(), col("secs_left").desc()
)

final_master_new = final_master_new.withColumn(
    "shots_taken",
    when(count("player_id").over(first_shot_window) == 1, lit(0)).otherwise(col("shots_taken"))
)

final_master_new = final_master_new.withColumn(
    "shots_made",
    when(count("player_id").over(first_shot_window) == 1, lit(0)).otherwise(col("shots_made"))
)

final_master_new = final_master_new.orderBy(
    col("shot_order").asc())

# COMMAND ----------

final_master_filter = final_master_new.filter(final_master.GP_Filter == "Yes")

# COMMAND ----------

final_master.count()

# COMMAND ----------

final_master_filter.count()

# COMMAND ----------

final_master_filter_sorted = final_master_filter.orderBy(
    col("game_id").asc(),
    col("quarter").asc(),
    col("mins_left").desc(),
    col("secs_left").desc()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Model Building**

# COMMAND ----------

master_model = final_master_filter.select("FULL NAME", "GAME_ID", "TEAM_ABBRV", "POSITION", "ACTION_TYPE", "SHOT_TYPE", "BASIC_ZONE", "ZONE_NAME", "ZONE_RANGE", "TS%", "eFG%", "2P%", "ORTG", "SHOT_CAT")

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

train_data = master_model.filter(col("game_id") <= 22100861)
test_data = master_model.filter(col("game_id") > 22100861)

# COMMAND ----------

lr = LogisticRegression(featuresCol="features", labelCol="SHOT_CAT")

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

model = pipeline.fit(train_data)

# COMMAND ----------

predictions = model.transform(test_data)

# predictions.select("SHOT_CAT", "probability", "prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Create xP Statistic**

# COMMAND ----------

predictions_new = predictions.select("FULL NAME", "GAME_ID", "TEAM_ABBRV", "POSITION", "SHOT_TYPE", "SHOT_CAT", "probability", "prediction")

# COMMAND ----------

df_with_prob_made = predictions_new.withColumn("prob_made", vector_to_array(col("probability"))[1])
predictions_clean = df_with_prob_made.select("FULL NAME", "GAME_ID", "TEAM_ABBRV", "SHOT_TYPE", "SHOT_CAT", "prob_made", "prediction")

# COMMAND ----------

df_withp_flag = predictions_clean.withColumn("shot_pt", when(col("SHOT_TYPE") == "2PT Field Goal", 2).otherwise(3))

# COMMAND ----------

df_pts = df_withp_flag.withColumn("pts", col("shot_pt") * col("shot_cat"))
df_xp = df_pts.withColumn("xP", col("shot_pt") * col("prob_made"))

# COMMAND ----------

master_xp = df_xp.select("FULL NAME", "GAME_ID", "TEAM_ABBRV", "SHOT_CAT", "prediction", "shot_pt", "prob_made", "xP", "pts")

# COMMAND ----------

# Group and aggregate
team_game_summary = master_xp.groupBy("GAME_ID", "TEAM_ABBRV") \
    .agg(
        _sum("xP").alias("total_xP"),
        _sum("pts").alias("total_pts")
    ) \
    .orderBy("GAME_ID", "TEAM_ABBRV")

# COMMAND ----------

game_summary = team_game_summary.withColumn("xP_performance", when(col("total_pts") > col("total_xP"), "yes").otherwise("no"))

# display(game_summary.limit(10))

# COMMAND ----------

performance_summary = game_summary.groupBy("TEAM_ABBRV").agg(
    sum(when(col("xP_performance") == "yes", 1).otherwise(0)).alias("outperform"),
    sum(when(col("xP_performance") == "no", 1).otherwise(0)).alias("underperform")
)

# COMMAND ----------

performance_summary_pct = performance_summary.withColumn("total_games", col("outperform") + col("underperform"))
performance_summary_pct = performance_summary_pct.withColumn("outperform_pct1", round(col("outperform") / col("total_games")*100))
performance_summary_pct = performance_summary_pct.withColumn("underperform_pct1", round(col("underperform") / col("total_games")*100))

# COMMAND ----------

performance_summary_pct_clean = performance_summary_pct.withColumn("outperform_pct", concat(col("outperform_pct1").cast("string"), lit("%")))
performance_summary_pct_clean = performance_summary_pct_clean.withColumn("underperform_pct", concat(col("underperform_pct1").cast("string"), lit("%")))

# COMMAND ----------

xp_summary = performance_summary_pct_clean.drop("total_games", "underperform_pct1", "outperform_pct1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Top 10 Teams - *Outperform* xP**

# COMMAND ----------

display(xp_summary.orderBy("outperform_pct", ascending=False).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Top 10 Teams - *Underperform* xP**

# COMMAND ----------

display(xp_summary.orderBy("underperform_pct", ascending=False).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Model Evaluation**

# COMMAND ----------

# # Accuracy Evaluator
# accuracy_evaluator = MulticlassClassificationEvaluator(
#     labelCol="SHOT_CAT", predictionCol="prediction", metricName="accuracy"
# )

# # Calculate accuracy
# accuracy = accuracy_evaluator.evaluate(predictions)
# print(f"Accuracy: {accuracy:.2f}")

# COMMAND ----------

# prediction_and_labels = predictions.select("prediction", "SHOT_CAT").rdd.map(lambda x: (float(x[0]), float(x[1])))

# # Compute confusion matrix
# metrics = MulticlassMetrics(prediction_and_labels)
# confusion_matrix = metrics.confusionMatrix().toArray()

# # Display confusion matrix
# print("Confusion Matrix:")
# print(confusion_matrix)

# COMMAND ----------

# auc_evaluator = BinaryClassificationEvaluator(
#     labelCol="SHOT_CAT", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
# )

# # Calculate AUC
# auc = auc_evaluator.evaluate(predictions)
# print(f"AUC: {auc:.2f}")

# COMMAND ----------

# # Extract the Logistic Regression model
# lr_model = model.stages[-1]

# # Get coefficients and intercept
# coefficients = lr_model.coefficients
# intercept = lr_model.intercept

# # List of feature names
# feature_names = numerical_cols + [f"{col}_encoded" for col in categorical_cols]

# # Display Feature Importance
# print(f"Intercept: {intercept}")
# print("Feature Importance (Coefficients):")
# for name, coef in zip(feature_names, coefficients):
#     print(f"{name}: {coef}")
