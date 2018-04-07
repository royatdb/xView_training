# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Cluster:
# MAGIC * DBR 4.0+ (CPU)
# MAGIC * python 3
# MAGIC * tensorflow==1.6.0
# MAGIC * tqdm==4.20.0
# MAGIC * Pillow==5.1.0
# MAGIC * scikit-image==0.13.1

# COMMAND ----------

# MAGIC %md
# MAGIC # Agenda:
# MAGIC * Create `xview_train_db.record` 
# MAGIC * Create `xview_test_db.record` 
# MAGIC * Create `xview_label_map.pbtxt` 

# COMMAND ----------

import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import argparse
import os
import json
import csv
from PIL import Image, ImageDraw
import skimage.filters as filters

# COMMAND ----------

def scale(x,range1=(0,0),range2=(0,0)):
    """
    Linear scaling for a value x
    """
    return range2[0]*(1 - (x-range1[0]) / (range1[1]-range1[0])) + range2[1]*((x-range1[0]) / (range1[1]-range1[0]))


def get_image(fname):    
    """
    Get an image from a filepath in ndarray format
    """
    return np.array(Image.open(fname))


def get_labels(fname):
    """
    Gets label data from a geojson label file

    Args:
        fname: file path to an xView geojson label file

    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'

    return coords, chips, classes


def boxes_from_coords(coords):
    """
    Processes a coordinate array from a geojson into (xmin,ymin,xmax,ymax) format

    Args:
        coords: an array of bounding box coordinates

    Output:
        Returns an array of shape (N,4) with coordinates in proper format
    """
    nc = np.zeros((coords.shape[0],4))
    for ind in range(coords.shape[0]):
        x1,x2 = coords[ind,:,0].min(),coords[ind,:,0].max()
        y1,y2 = coords[ind,:,1].min(),coords[ind,:,1].max()
        nc[ind] = [x1,y1,x2,y2]
    return nc


def chip_image(img,coords,classes,shape=(300,300)):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.

    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    height,width,_ = img.shape
    wn,hn = shape
    
    w_num,h_num = (int(width/wn),int(height/hn))
    images = np.zeros((w_num*h_num,hn,wn,3))
    total_boxes = {}
    total_classes = {}
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or( np.logical_and((coords[:,0]<((i+1)*wn)),(coords[:,0]>(i*wn))),
                               np.logical_and((coords[:,2]<((i+1)*wn)),(coords[:,2]>(i*wn))))
            out = coords[x]
            y = np.logical_or( np.logical_and((out[:,1]<((j+1)*hn)),(out[:,1]>(j*hn))),
                               np.logical_and((out[:,3]<((j+1)*hn)),(out[:,3]>(j*hn))))
            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i),0,wn),
                                          np.clip(outn[:,1]-(hn*j),0,hn),
                                          np.clip(outn[:,2]-(wn*i),0,wn),
                                          np.clip(outn[:,3]-(hn*j),0,hn))))
            box_classes = classes[x][y]
            
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8),total_boxes,total_classes

# COMMAND ----------

def to_tf_example(img, boxes, class_num):
    """
    Converts a single image with respective boxes into a TFExample.  Multiple TFExamples make up a TFRecord.

    Args:
        img: an image array
        boxes: an array of bounding boxes for the given image
        class_num: an array of class numbers for each bouding box

    Output:
        A TFExample containing encoded image data, scaled bounding boxes with classes, and other metadata.
    """
    encoded = convertToJpeg(img)

    width = img.shape[0]
    height = img.shape[1]

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    
    for ind,box in enumerate(boxes):
        xmin.append(box[0] / width)
        ymin.append(box[1] / height)
        xmax.append(box[2] / width)
        ymax.append(box[3] / height) 
        classes.append(int(class_num[ind]))

    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/encoded': bytes_feature(encoded),
            'image/format': bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/label': int64_list_feature(classes),
    }))
    
    return example

def convertToJpeg(im):
    """
    Converts an image array into an encoded JPEG string.

    Args:
        im: an image array

    Output:
        an encoded byte string containing the converted JPEG image.
    """
    with io.BytesIO() as f:
        im = Image.fromarray(im)
        im.save(f, format='JPEG')
        return f.getvalue()

def create_tf_record(output_filename, images, boxes):
    """ DEPRECIATED
    Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        images: an array of images to create a record for
        boxes: an array of bounding box coordinates ([xmin,ymin,xmax,ymax]) with the same index as images
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    k = 0
    for idx, image in enumerate(images):
        if idx % 100 == 0:
            print('On image %d of %d' %(idx, len(images)))

        tf_example = to_tf_example(image,boxes[idx],fname)
        if np.array(tf_example.features.feature['image/object/bbox/xmin'].float_list.value[0]).any():
            writer.write(tf_example.SerializeToString())
            k = k + 1
    
    print("saved: %d chips" % k)
    writer.close()

## VARIOUS HELPERS BELOW ##

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# COMMAND ----------

def rotate_image_and_boxes(img, deg, pivot, boxes):
    """
    Rotates an image and corresponding bounding boxes.  Bounding box rotations are kept axis-aligned,
        so multiples of non 90-degrees changes the area of the bounding box.

    Args:
        img: the image to be rotated in array format
        deg: an integer representing degree of rotation
        pivot: the axis of rotation. By default should be the center of an image, but this can be changed.
        boxes: an (N,4) array of boxes for the image

    Output:
        Returns the rotated image array along with correspondingly rotated bounding boxes
    """

    if deg < 0:
        deg = 360-deg
    deg = int(deg)
        
    angle = 360-deg
    padX = [img.shape[0] - pivot[0], pivot[0]]
    padY = [img.shape[1] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant').astype(np.uint8)
    #scipy ndimage rotate takes ~.7 seconds
    #imgR = ndimage.rotate(imgP, angle, reshape=False)
    #PIL rotate uses ~.01 seconds
    imgR = Image.fromarray(imgP).rotate(angle)
    imgR = np.array(imgR)
    
    theta = deg * (np.pi/180)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #  [(cos(theta), -sin(theta))] DOT [xmin, xmax] = [xmin*cos(theta) - ymin*sin(theta), xmax*cos(theta) - ymax*sin(theta)]
    #  [sin(theta), cos(theta)]        [ymin, ymax]   [xmin*sin(theta) + ymin*cos(theta), xmax*cos(theta) + ymax*cos(theta)]

    newboxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        #The 'x' values are not centered by the x-center (shape[0]/2)
        #but rather the y-center (shape[1]/2)
        
        xmin -= pivot[1]
        xmax -= pivot[1]
        ymin -= pivot[0]
        ymax -= pivot[0]

        bfull = np.array([ [xmin,xmin,xmax,xmax] , [ymin,ymax,ymin,ymax]])
        c = np.dot(R,bfull) 
        c[0] += pivot[1]
        c[0] = np.clip(c[0],0,img.shape[1])
        c[1] += pivot[0]
        c[1] = np.clip(c[1],0,img.shape[0])
        
        if np.all(c[1] == img.shape[0]) or np.all(c[1] == 0):
            c[0] = [0,0,0,0]
        if np.all(c[0] == img.shape[1]) or np.all(c[0] == 0):
            c[1] = [0,0,0,0]

        newbox = np.array([np.min(c[0]),np.min(c[1]),np.max(c[0]),np.max(c[1])]).astype(np.int64)

        if not (np.all(c[1] == 0) and np.all(c[0] == 0)):
            newboxes.append(newbox)
    
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]], newboxes

def shift_image(image,bbox):
    """
    Shift an image by a random amount on the x and y axis drawn from discrete  
        uniform distribution with parameter min(shape/10)

    Args:
        image: the image to be shifted in array format
        bbox: an (N,4) array of boxes for the image

    Output:
        The shifted image and corresponding boxes
    """
    shape = image.shape[:2]
    maxdelta = min(shape)/10
    dx,dy = np.random.randint(-maxdelta,maxdelta,size=(2))
    newimg = np.zeros(image.shape,dtype=np.uint8)
    
    nb = []
    for box in bbox:
        xmin,xmax = np.clip((box[0]+dy,box[2]+dy),0,shape[1])
        ymin,ymax = np.clip((box[1]+dx,box[3]+dx),0,shape[0])

        #we only add the box if they are not all 0
        if not(xmin==0 and xmax ==0 and ymin==0 and ymax ==0):
            nb.append([xmin,ymin,xmax,ymax])
    
    newimg[max(dx,0):min(image.shape[0],image.shape[0]+dx),
           max(dy,0):min(image.shape[1],image.shape[1]+dy)] = \
    image[max(-dx,0):min(image.shape[0],image.shape[0]-dx),
          max(-dy,0):min(image.shape[1],image.shape[1]-dy)]
    
    return newimg, nb

def salt_and_pepper(img,prob=.005):
    """
    Applies salt and pepper noise to an image with given probability for both.

    Args:
        img: the image to be augmented in array format
        prob: the probability of applying noise to the image

    Output:
        Augmented image
    """

    newimg = np.copy(img)
    whitemask = np.random.randint(0,int((1-prob)*200),size=img.shape[:2])
    blackmask = np.random.randint(0,int((1-prob)*200),size=img.shape[:2])
    newimg[whitemask==0] = 255
    newimg[blackmask==0] = 0
        
    return newimg


def gaussian_blur(img, max_sigma=1.5):
    """
    Use a gaussian filter to blur an image

    Args:
        img: image to be augmented in array format
        max_sigma: the maximum variance for gaussian blurring

    Output:
        Augmented image
    """
    return filters.gaussian(img,np.random.random()*max_sigma,multichannel=True)*255

def draw_bboxes(img,boxes):
    """
    A helper function to draw bounding box rectangles on images

    Args:
        img: image to be drawn on in array format
        boxes: An (N,4) array of bounding boxes

    Output:
        Image with drawn bounding boxes
    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    idx = 0

    for b in boxes:
        xmin,ymin,xmax,ymax = b
        
        for j in range(3):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source

# COMMAND ----------

# MAGIC %md
# MAGIC # Gathers and chips all images within a given folder at a given resolution.
# MAGIC * Args:
# MAGIC   * coords: an array of bounding box coordinates
# MAGIC   * chips: an array of filenames that each coord/class belongs to.
# MAGIC   * classes: an array of classes for each bounding box
# MAGIC   * folder_names: a list of folder names containing images
# MAGIC   * res: an (X,Y) tuple where (X,Y) are (width,height) of each chip respectively
# MAGIC * Output:
# MAGIC   * images, boxes, classes arrays containing chipped images, bounding boxes, and classes, respectively.

# COMMAND ----------

def get_images_from_filename_array(coords,chips,classes,folder_names,res=(250,250)):
    images =[]
    boxes = []
    clses = []

    k = 0
    bi = 0   
    
    for folder in folder_names:
        fnames = glob.glob(folder + "*.tif")
        fnames.sort()
        for fname in tqdm(fnames):
            #Needs to be "X.tif" ie ("5.tif")
            name = fname.split("\\")[-1]
            arr = get_image(fname)
            
            img,box,cls = chip_image(arr,coords[chips==name],classes[chips==name],res)

            for im in img:
                images.append(im)
            for b in box:
                boxes.append(b)
            for c in cls:
                clses.append(cls)
            k = k + 1
            
    return images, boxes, clses

# COMMAND ----------

# MAGIC %md
# MAGIC # Shuffles images, boxes, and classes, while keeping relative matching indices
# MAGIC * Args:
# MAGIC   * im: an array of images
# MAGIC   * box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
# MAGIC   * cls: an array of classes
# MAGIC * Output:
# MAGIC   * Shuffle image, boxes, and classes arrays, respectively

# COMMAND ----------

def shuffle_images_and_boxes_classes(im,box,cls):
    assert len(im) == len(box)
    assert len(box) == len(cls)
    
    perm = np.random.permutation(len(im))
    out_b = {}
    out_c = {}
    
    k = 0 
    for ind in perm:
        out_b[k] = box[ind]
        out_c[k] = cls[ind]
        k = k + 1
    return im[perm], out_b, out_c

# COMMAND ----------

image_folder = "/dbfs/home/tim/demo-dataset/xview/train_images/"
json_filepath = "/dbfs/home/tim/demo-dataset/xview/xView_train.geojson"
test_percent=0.2
suffix="db"
AUGMENT = True

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate TFRecord for `train` and `test`

# COMMAND ----------

res = [(300,300)]

SAVE_IMAGES = False
images = {}
boxes = {}
train_chips = 0
test_chips = 0

#Parameters
max_chips_per_res = 100000
train_writer = tf.python_io.TFRecordWriter("xview_train_%s.record" % suffix)
test_writer = tf.python_io.TFRecordWriter("xview_test_%s.record" % suffix)

coords,chips,classes = get_labels(json_filepath)

for res_ind, it in enumerate(res):
    tot_box = 0
    print("Res: %s" % str(it))
    ind_chips = 0

    fnames = glob.glob(image_folder + "*.tif")
    fnames.sort()

    for fname in tqdm(fnames):
        name = fname.split("/")[-1]
        arr = get_image(fname)

        im,box,classes_final = chip_image(arr,coords[chips==name],classes[chips==name],it)

        #Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
        im,box,classes_final = shuffle_images_and_boxes_classes(im,box,classes_final)
        split_ind = int(im.shape[0] * test_percent)

        for idx, image in enumerate(im):
            tf_example = to_tf_example(image,box[idx],classes_final[idx])

            #Check to make sure that the TF_Example has valid bounding boxes.  
            #If there are no valid bounding boxes, then don't save the image to the TFRecord.
            float_list_value = tf_example.features.feature['image/object/bbox/xmin'].float_list.value

            if (ind_chips < max_chips_per_res and np.array(float_list_value).any()):
                tot_box+=np.array(float_list_value).shape[0]

                if idx < split_ind:
                    test_writer.write(tf_example.SerializeToString())
                    test_chips+=1
                else:
                    train_writer.write(tf_example.SerializeToString())
                    train_chips += 1

                ind_chips +=1

                #Make augmentation probability proportional to chip size.  Lower chip size = less chance.
                #This makes the chip-size imbalance less severe.
                prob = np.random.randint(0,np.max(res))
                #for 200x200: p(augment) = 200/500 ; for 300x300: p(augment) = 300/500 ...

                if AUGMENT and prob < it[0]:

                    for extra in range(3):
                        center = np.array([int(image.shape[0]/2),int(image.shape[1]/2)])
                        deg = np.random.randint(-10,10)
                        #deg = np.random.normal()*30
                        newimg = salt_and_pepper(gaussian_blur(image))

                        #.3 probability for each of shifting vs rotating vs shift(rotate(image))
                        p = np.random.randint(0,3)
                        if p == 0:
                            newimg,nb = shift_image(newimg,box[idx])
                        elif p == 1:
                            newimg,nb = rotate_image_and_boxes(newimg,deg,center,box[idx])
                        elif p == 2:
                            newimg,nb = rotate_image_and_boxes(newimg,deg,center,box[idx])
                            newimg,nb = shift_image(newimg,nb)


                        newimg = (newimg).astype(np.uint8)

                        if idx%1000 == 0 and SAVE_IMAGES:
                            Image.fromarray(newimg).save('process/img_%s_%s_%s.png'%(name,extra,it[0]))

                        if len(nb) > 0:
                            tf_example = to_tf_example(newimg,nb,classes_final[idx])

                            #Don't count augmented chips for chip indices
                            if idx < split_ind:
                                test_writer.write(tf_example.SerializeToString())
                                test_chips += 1
                            else:
                                train_writer.write(tf_example.SerializeToString())
                                train_chips+=1
                        else:
                            if SAVE_IMAGES:
                                draw_bboxes(newimg,nb).save('process/img_nobox_%s_%s_%s.png'%(name,extra,it[0]))
    if res_ind == 0:
        max_chips_per_res = int(ind_chips * 1.5)
        print("Max chips per resolution: %s " % max_chips_per_res)

    print("Tot Box: %d" % tot_box)
    print("Chips: %d" % ind_chips)

print("saved: %d train chips" % train_chips)
print("saved: %d test chips" % test_chips)
train_writer.close()
test_writer.close() 

# COMMAND ----------

dbutils.fs.mkdirs("/mnt/roy/xview_train")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls .

# COMMAND ----------

# MAGIC %sh
# MAGIC cp *.record /dbfs/mnt/roy/xview_train

# COMMAND ----------

display(
  dbutils.fs.ls("/mnt/roy/xview_train")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create `xview_label_map.pbtxt`

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/DIUx-xView/data_utilities/master/xview_class_labels.txt

# COMMAND ----------

# MAGIC %sh
# MAGIC cat xview_class_labels.txt

# COMMAND ----------

# MAGIC %sh
# MAGIC sed -e "s/\:/\tname: / ; s/^/item \{\tid: / ; s/$/'\n}/ ; s/\t/\n\t/; s/\}/\}\n/; s/name\: /\n\tname\: '/" xview_class_labels.txt > xview_label_map.pbtxt

# COMMAND ----------

# MAGIC %sh
# MAGIC cat xview_label_map.pbtxt

# COMMAND ----------

# MAGIC %sh
# MAGIC cp xview_label_map.pbtxt /dbfs/mnt/roy/xview_train

# COMMAND ----------

# MAGIC %md
# MAGIC # Files needed for training:

# COMMAND ----------

display(
  dbutils.fs.ls("/mnt/roy/xview_train")
)

# COMMAND ----------

