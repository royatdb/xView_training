# Databricks notebook source
# MAGIC %sh
# MAGIC ls -lh ./train

# COMMAND ----------

dbutils.tensorboard.start("./train")

# COMMAND ----------

