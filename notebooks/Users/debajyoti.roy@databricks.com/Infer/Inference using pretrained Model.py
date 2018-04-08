# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Cluster:
# MAGIC * DBR 4.0+ (CPU)
# MAGIC * python 3
# MAGIC * tensorflow==1.6.0
# MAGIC * tqdm==4.20.0
# MAGIC * Pillow==5.1.0

# COMMAND ----------

# MAGIC %md
# MAGIC # Images

# COMMAND ----------

display(
  dbutils.fs.ls("home/tim/demo-dataset/xview/train_images/")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-trained Model checkpoints

# COMMAND ----------

display(
  dbutils.fs.ls("/mnt/roy/xview_model/public_release/")
)

# COMMAND ----------

display(
  dbutils.fs.ls("/mnt/roy/xview_fine_tuned")
)

# COMMAND ----------

import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

def chip_image(img, chip_size=(300,300)):
    width,height,_ = img.shape
    wn,hn = chip_size
    images = np.zeros((int(width/wn) * int(height/hn),wn,hn,3))
    k = 0
    for i in tqdm(range(int(width/wn))):
        for j in range(int(height/hn)):
            
            chip = img[wn*i:wn*(i+1),hn*j:hn*(j+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8)

# COMMAND ----------

# checkpoint = "/dbfs/mnt/roy/xview_model/public_release/vanilla.pb"
checkpoint = "/dbfs/mnt/roy/xview_fine_tuned/frozen_inference_graph.pb"
chip_size = 300

# COMMAND ----------

# MAGIC %md
# MAGIC # Input image

# COMMAND ----------

input = "/dbfs/home/tim/demo-dataset/xview/train_images/1192.tif"

# COMMAND ----------

fig = plt.figure()
input_img = Image.open(input)
plt.imshow(input_img)
display(fig)

# COMMAND ----------

arr = np.array(input_img)
chip_size = (chip_size,chip_size)
images = chip_image(arr,chip_size)
print(images.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # Object Detection

# COMMAND ----------

def generate_detections(checkpoint,images):
    
    print("Creating Graph...")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    boxes = []
    scores = []
    classes = []
    k = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_np in tqdm(images):
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                score = detection_graph.get_tensor_by_name('detection_scores:0')
                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                boxes.append(box)
                scores.append(score)
                classes.append(clss)
                
    boxes =   np.squeeze(np.array(boxes))
    scores = np.squeeze(np.array(scores))
    classes = np.squeeze(np.array(classes))

    return boxes,scores,classes

# COMMAND ----------

boxes, scores, classes = generate_detections(checkpoint,images)

#Process boxes to be full-sized
width,height,_ = arr.shape
cwn,chn = (chip_size)
wn,hn = (int(width/cwn),int(height/chn))

num_preds = 250
bfull = boxes[:wn*hn].reshape((wn,hn,num_preds,4))
b2 = np.zeros(bfull.shape)
b2[:,:,:,0] = bfull[:,:,:,1]
b2[:,:,:,1] = bfull[:,:,:,0]
b2[:,:,:,2] = bfull[:,:,:,3]
b2[:,:,:,3] = bfull[:,:,:,2]

bfull = b2
bfull[:,:,:,0] *= cwn
bfull[:,:,:,2] *= cwn
bfull[:,:,:,1] *= chn
bfull[:,:,:,3] *= chn
for i in range(wn):
    for j in range(hn):
        bfull[i,j,:,0] += j*cwn
        bfull[i,j,:,2] += j*cwn

        bfull[i,j,:,1] += i*chn
        bfull[i,j,:,3] += i*chn

bfull = bfull.reshape((hn*wn,num_preds,4))

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictions:

# COMMAND ----------

def draw_bboxes(img,boxes,classes):
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    idx = 0

    for i in range(len(boxes)):
        xmin,ymin,xmax,ymax = boxes[i]
        c = classes[i]

        draw.text((xmin+15,ymin+15), str(c))

        for j in range(4):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source

# COMMAND ----------

bs = bfull[scores > .5]
cs = classes[scores>.5]
s = input.split("/")[::-1]
out = draw_bboxes(arr,bs,cs)
fig = plt.figure()
plt.imshow(out)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC # Inferences using `Spark`

# COMMAND ----------

dbutils.fs.mkdirs("mnt/roy/xview_out")

# COMMAND ----------

with open("/dbfs/mnt/roy/xview_out/1192_prediction.txt",'w') as f:
  for i in range(bfull.shape[0]):
    for j in range(bfull[i].shape[0]):
      box = bfull[i,j]
      class_prediction = classes[i,j]
      score_prediction = scores[i,j]
      f.write('%d,%d,%d,%d,%d,%f\n' % (box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

classes = spark.read.csv("/mnt/roy/xview_classes/xview_class_labels.txt", sep=":", inferSchema=True).\
  withColumn("id", col("_c0")).\
  withColumn("name", col("_c1")).\
  drop("_c0", "_c1")
display(classes)

# COMMAND ----------

predictions = spark.read.csv("/mnt/roy/xview_out/1192_prediction.txt", sep=",", inferSchema=True).\
  withColumn("class_id", col("_c4")).\
  withColumn("score", col("_c5")).\
  drop("_c4", "_c5")
display(predictions)

# COMMAND ----------

display(
  predictions.join(classes, predictions.class_id==classes.id).groupBy("name").count().orderBy("count").filter("count>0")
)

# COMMAND ----------

