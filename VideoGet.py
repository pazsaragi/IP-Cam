import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2
import datetime
import random
import logging
import time
import concurrent.futures
from random import randint
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from ipcam_utils import *
from utils import label_map_util
from utils import visualization_utils_changed as vis_util
from threading import Thread

#####################################################################################
#GLOBALS

sys.path.append("..")

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

im_width = 1280
im_height = 720
frame_counter = 0
threshold_frame = 90
move_cam = False
has_moved = False
NUM_CLASSES = 90
find_or_not = False
sampling_counter = 0
delay_time = 0.02

#####################################################################################
#FPS

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

#####################################################################################
#RETURN FRAME

class VideoGet(object):
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=(), daemon=True).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

#####################################################################################
#SHOW FRAME/BBOX/MOVE CAM

class VideoShow(object):
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None, detection_graph=None, ptz=None):
        self.frame = frame
        self.stopped = False
        self.detection_graph = detection_graph
        self.ptz = ptz

    def start(self):
        Thread(target=self.show, args=(), daemon=True).start()
        return self

    def show(self):
        frame_counter = 0
        left, right, top, bottom = 0, 0, 0, 0
        while not self.stopped:
          frame_counter += 1
          
          cv2.imshow("Video", self.frame)
          #if frame_counter % 1 == 0:
          if not self.stopped:
            _, left, right, top, bottom = make_pred(self.frame, detection_graph)
            #An important delay because the camera will move around too much
            if frame_counter % 2 == 0:
                self.start_moving(left, right, top, bottom)
          self.draw(left, right, top, bottom)
          if cv2.waitKey(1) == ord("q"):
              self.stopped = True

    def stop(self):
        self.stopped = True

    def draw(self, left, right, top, bottom):
      cv2.rectangle(self.frame, (int(left),int(top)),(int(right),int(bottom)),(255, 0, 0), 3)

    def start_moving(self, left, right, top, bottom):
        Thread(target=self.center, args=(left, right, top, bottom), daemon=True).start()
        return self

    def center(self, left, right, top, bottom):
        """
            -----top-----
            |           |
        left|           |right
            |           |
            ----bottom--- 
            bottom =  ymax * im_height
            top = ymin * im_height
            right = xmax * im_width
            left = xmin * im_width
        """
        centered = True
        #width = np.abs(left - right )
        #height = np.abs(top - bottom )
        #area = height * width
        logging.info("Thread %s: starting")

        if right > (im_width * 0.85):  
            print("Going right : \nright: ",right, (im_width * 0.9), '******\n')
            ptz.pan(Direction='right', Steps=1, Speed=1)
            time.sleep(delay_time)
            ptz.pan(Direction='stop', Steps=1, Speed=1)
            print("Movement has finished.")
            centered = False

        if left < (im_width * 0.15):
            print("Going left : \nleft: ", left, (im_width * 0.1), '******\n')
            ptz.pan(Direction='left', Steps=1, Speed=1)
            time.sleep(delay_time)
            ptz.pan(Direction='stop', Steps=1, Speed=1)
            print("Movement has finished.")
            centered = False

        #Currently not using Up/Down since Bounding Boxes are sensitive to these
        # if bottom > (im_height * 0.9):
        #   print("Going Down : \ntop: ", bottom, (im_height * 0.9), '\n******\n')
        #   ptz.pan(Direction='down', Steps=1, Speed=1)
        #   time.sleep(delay_time)
        #   ptz.pan(Direction='stop', Steps=1, Speed=1)
        #   has_completed_once = True
        #   print("Movement has finished.")
        #   centered = False

        # if top < (im_height * 0.1):#If top is lower than bottom threshold go up
        #   print("Going Up : \ntop: ", top, (im_height * 0.9), '\n******\n')
        #   ptz.pan(Direction='up', Steps=1, Speed=1)
        #   time.sleep(delay_time)
        #   ptz.pan(Direction='stop', Steps=1, Speed=1)
        #   has_completed_once = True
        #   print("Movement has finished.")
        #   centered = False

        #Zoom functionality does not work on this camera
        # if area < too_small_box:
        #   print("Zooming in: \n")
        #   ptz.pan(Direction='zoomin', Steps=1, Speed=1)
        #   time.sleep(delay_time)
        #   has_completed_once = True
        #   print("Movement has finished.")
        #   centered = False

        # if area > too_big_box:
        #   print("Zooming out: \n")
        #   ptz.pan(Direction='zoomout', Steps=1, Speed=1)
        #   time.sleep(delay_time)
        #   has_completed_once = True
        #   print("Movement has finished.")
        #   centered = False

        if centered == True:
            print("The person is in frame.\n")
            has_completed_once = True

#####################################################################################
#CREATE PTZ OBJECT

def setup_ptz(Camera_IP, Camera_User, Camera_PW):
  print("Loading Camera and setting up Pan-Tilt-Zoom commands...")
  ptz = PTZ_commands(Camera_User, Camera_PW, Camera_IP)
  return ptz

#####################################################################################
#OBJECT DETECTION PREDICTION

def make_pred(image_np, detection_graph):
  start_time = time.time()
  
  flag = False
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  # Each box represents a part of the image where a particular object was detected.
  boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  # Each score represent how level of confidence for each of the objects.
  # Score is shown on the result image, together with the class label.
  scores = detection_graph.get_tensor_by_name('detection_scores:0')
  classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  # Actual detection.
  (boxes, scores, classes, num_detections) = sess.run(
    [boxes, scores, classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})
  time_diff = round(time.time() - start_time)
  print("It took:", (time_diff % 60), "seconds, to run a prediction using", MODEL_NAME)

  #If the Person score is above some threshold and is the best class then compute bbox and move
  if scores[0][0] > 0.5 and np.argmax(scores) == 0 :
    print("Person found...")
    flag = False
    # Visualization of the results of a detection.
    _, ymin, xmin, ymax, xmax = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    return flag, left, right, top, bottom

  else:

    print("No Person found searching for humans...")
    left, right, top, bottom = 0, 0, 0, 0
    flag = True

    return flag, left, right, top, bottom

#####################################################################################
#RANDOM MOVEMENT
#Borrowed from ipcam_utils.py 

def findPerson(ptz):

  has_completed_once = False
  random_move = randint(1,6)

  if random_move == 1:
    print("Going right\n")
    ptz.pan(Direction='right', Steps=1, Speed=1)
    time.sleep(delay_time)
    ptz.pan(Direction='stop', Steps=1, Speed=1)
    has_completed_once = True
    print("Movement has finished.")

  if random_move == 2:
    print("Going left\n")
    ptz.pan(Direction='left', Steps=1, Speed=1)
    time.sleep(delay_time)
    ptz.pan(Direction='stop', Steps=1, Speed=1)
    has_completed_once = True
    print("Movement has finished.")

  if random_move == 3:
    print("Going Down\n")
    ptz.pan(Direction='down', Steps=1, Speed=1)
    time.sleep(delay_time)
    ptz.pan(Direction='stop', Steps=1, Speed=1)
    has_completed_once = True
    print("Movement has finished.")

  if random_move == 4:
    print("Going Up\n")
    ptz.pan(Direction='up', Steps=1, Speed=1)
    time.sleep(delay_time)
    ptz.pan(Direction='stop', Steps=1, Speed=1)
    has_completed_once = True
    print("Movement has finished.")

  if random_move == 5:
    print("Horizontal Scan\n")
    ptz.pan(Direction='hscan', Steps=1, Speed=1)
    time.sleep(delay_time)
    ptz.pan(Direction='stop', Steps=1, Speed=1)
    has_completed_once = True
    print("Movement has finished.")

  if random_move == 6:
    print("Vertical Scan\n")
    ptz.pan(Direction='vscan', Steps=1, Speed=1)
    time.sleep(delay_time)
    ptz.pan(Direction='stop', Steps=1, Speed=1)
    has_completed_once = True
    print("Movement has finished.")

  return has_completed_once

#####################################################################################
#Loads model from TF OBJECT DETECTION API

def open_model(Model_name):
  print("Opening....", Model_name)
  MODEL_FILE = MODEL_NAME + '.tar.gz'
  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

#####################################################################################
#THREADER 

def threadedBoth(ptz, detection_graph):
  """
  This thread is dedicated to getting video frames.
  """
  video_getter = VideoGet(ptz.rtsp).start()
  video_show = VideoShow(video_getter.frame, detection_graph, ptz).start()

  left, right, top, bottom = 0,0,0,0
  while True:
    if video_getter.stopped or video_show.stopped:
      video_show.stop()
      video_getter.stop()
      break
    frame = video_getter.frame
    video_show.frame = frame

#####################################################################################
#Load Model/ setup PTZ

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
  parser.add_argument('--Model_exists', type=bool, default=True)
  parser.add_argument('--Model_name',   type=str,   help='Name of the model you want to use.', default="ssd_mobilenet_v1_coco_11_06_2017")
  parser.add_argument('--Camera_IP',    type=str, required=True)
  parser.add_argument('--Camera_User',  type=str, required=True)
  parser.add_argument('--Camera_PW',    type=str, required=True)
  args = parser.parse_args()

  MODEL_NAME = args.Model_name
  if args.Model_exists == False:
    open_model(MODEL_NAME)

  ptz = setup_ptz(args.Camera_IP, args.Camera_User, args.Camera_PW)

#####################################################################################
#Load Graph

  start = time.time()
  print("Setting up graph....")
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
  time_diff = round(time.time() - start)
  print( "It took:", (time_diff % 60), "seoncds, to load graph and label map.")

#####################################################################################
#Begin prediction/threading

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      fps = FPS().start()
      threadedBoth(ptz, detection_graph)

      fps.stop()
      print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))          
      print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
      cap.release()
      sess.close()

#####################################################################################