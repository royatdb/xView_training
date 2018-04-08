# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Cluster:
# MAGIC * DBR 4.0+ (CPU)
# MAGIC * python 3

# COMMAND ----------

# MAGIC %md
# MAGIC # Agenda:
# MAGIC * Download `model.ckpt`
# MAGIC * Create `pipeline.config 

# COMMAND ----------

# MAGIC %md
# MAGIC # From [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): 
# MAGIC * Download COCO-trained model for `faster_rcnn_nas` (mAP = 43, Speed = 1833 ms)

# COMMAND ----------

# MAGIC %sh
# MAGIC wget http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/mnt/roy/xview_train_model")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -hl .

# COMMAND ----------

import tarfile,sys

# COMMAND ----------

tar = tarfile.open("./faster_rcnn_nas_coco_2018_01_28.tar.gz")
tar.extractall()
tar.close()

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -lh faster_rcnn_nas_coco_2018_01_28

# COMMAND ----------

# MAGIC %sh
# MAGIC cp ./faster_rcnn_nas_coco_2018_01_28/model.ckpt.* /dbfs/mnt/roy/xview_train_model

# COMMAND ----------

display(
  dbutils.fs.ls("/mnt/roy/xview_train_model")
)

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/faster_rcnn_nas_coco.config

# COMMAND ----------

# MAGIC %md
# MAGIC # Download config file for `Faster R-CNN with NASNet-A`
# MAGIC from [sample configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

# COMMAND ----------

# MAGIC %sh cat faster_rcnn_nas_coco.config

# COMMAND ----------

# MAGIC %sh 
# MAGIC sed -e "s/PATH_TO_BE_CONFIGURED\/model.ckpt/\/dbfs\/mnt\/roy\/xview_train_model\/model.ckpt/ ; s/PATH_TO_BE_CONFIGURED\/mscoco_train.record/\/dbfs\/mnt\/roy\/xview_train\/xview_train_db.record/ ; s/PATH_TO_BE_CONFIGURED\/mscoco_val.record/\/dbfs\/mnt\/roy\/xview_train\/xview_test_db.record/ ; s/PATH_TO_BE_CONFIGURED\/mscoco_label_map.pbtxt/\/dbfs\/mnt\/roy\/xview_train\/xview_label_map.pbtxt/" faster_rcnn_nas_coco.config > pipeline.config

# COMMAND ----------

# MAGIC %sh cat pipeline.config

# COMMAND ----------

# MAGIC %sh
# MAGIC cp pipeline.config /dbfs/mnt/roy/xview_train

# COMMAND ----------

# MAGIC %md
# MAGIC # `pipeline.config`

# COMMAND ----------

display(
  dbutils.fs.ls("/mnt/roy/xview_train")
)

# COMMAND ----------

