import cv2
import numpy as np
import tkinter.messagebox
from tkinter import *
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from io import StringIO
from PIL import Image
from utils import label_map_util
from timeit import default_timer as timer
from utils import visualization_utils as vis_util
rt = Tk()
rt.withdraw()
thresh = 05.00
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


cap = cv2.VideoCapture(0)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
      ret,image_np = cap.read()
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      x = [category_index.get(i) for i in classes[0]]
      y = x[1]
      
      if (y['name'] == 'person'):
         start = timer()
         
        
         if (start > thresh):
            
            choice = tkinter.messagebox.askquestion('What You want to do?')
            
            if choice == 'yes':
                tkinter.messagebox.showinfo('Notification','This can be danger')
                cv2.destroyAllWindows()
                cap.release()
                break
            elif choice == 'no':
                thresh = thresh+10
                continue


      
      cv2.imshow('image',cv2.resize(image_np,(1280,960)))
      if cv2.waitKey(25) & 0xFF == ord('p'):
          cv2.destroyAllWindows()
          cap.release()
          break

      





