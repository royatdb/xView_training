# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Cluster:
# MAGIC * DBR 4.0+ (CPU)
# MAGIC * python 3

# COMMAND ----------

# MAGIC %md
# MAGIC # Train:
# MAGIC * Setup `Tensorflow Object Detection`
# MAGIC * Train
# MAGIC * Export model checkpoint

# COMMAND ----------

# MAGIC %md
# MAGIC # clone `Tensorflow Object Detection`

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/tensorflow/models.git

# COMMAND ----------

# MAGIC %sh
# MAGIC cat models/research/object_detection/utils/learning_schedules.py

# COMMAND ----------

# MAGIC %sh sed -i 's/range(num_boundaries)/list(range(num_boundaries))/' models/research/object_detection/utils/learning_schedules.py

# COMMAND ----------

# MAGIC %sh
# MAGIC cat models/research/object_detection/utils/learning_schedules.py

# COMMAND ----------

dbutils.fs.put("/tmp/finetune_object_detection.sh", """
#!/bin/bash

apt-get -y install protobuf-compiler python-pil python-lxml python-tk
cd ./models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
curl https://bootstrap.pypa.io/get-pip.py | python3
python3 -m pip install tensorflow
python3 -m pip install Cython
python3 -m pip install jupyter
python3 -m pip install matplotlib
python3 setup.py build
python3 setup.py install
python3 object_detection/builders/model_builder_test.py

cd object_detection
python3 train.py --logtostderr --train_dir=../../../train --pipeline_config_path=/dbfs/mnt/roy/xview_train/pipeline.config 

""", True)

# COMMAND ----------

# MAGIC %sh cp /dbfs/tmp/finetune_object_detection.sh .

# COMMAND ----------

# MAGIC %sh chmod 777 finetune_object_detection.sh

# COMMAND ----------

# MAGIC %sh cat finetune_object_detection.sh

# COMMAND ----------

# MAGIC %sh ./finetune_object_detection.sh

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -ls ./train

# COMMAND ----------

# MAGIC %md
# MAGIC # Export model checkpoint

# COMMAND ----------

dbutils.fs.put("/tmp/fine_tuned_model.sh", """
#!/bin/bash

cd ./models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd object_detection
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path /dbfs/mnt/roy/xview_train/pipeline.config --trained_checkpoint_prefix ../../../train/model.ckpt-207 --output_directory ../../../fine_tuned_model


""", True)

# COMMAND ----------

# MAGIC %sh cp /dbfs/tmp/fine_tuned_model.sh .

# COMMAND ----------

# MAGIC %sh chmod 777 fine_tuned_model.sh

# COMMAND ----------

# MAGIC %sh ./fine_tuned_model.sh

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lh ./fine_tuned_model

# COMMAND ----------

dbutils.fs.mkdirs("/mnt/roy/xview_fine_tuned")

# COMMAND ----------

# MAGIC %sh
# MAGIC cp ./fine_tuned_model/frozen_inference_graph.pb /dbfs/mnt/roy/xview_fine_tuned

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lh /dbfs/mnt/roy/xview_fine_tuned

# COMMAND ----------

