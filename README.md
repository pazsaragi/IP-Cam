# IP Camera

These scripts are used to send HTTP commands to an IP Camera. These commands will focus an object (using Tensorflow Object Detection repo) to place a subject in the center of a frame. Place script files in the object detection directory. 


Requires:

Tensorflow 1.12.0
Anaconda
OpenCV2

Installation Instructions:

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b


Clone the model repo from tensorflow https://github.com/tensorflow/models

Conda create -n "env name" python=X.X anaconda https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

Conda install tensorflow
pip install other dependencies listed in the above
Install coco api by git clone https://github.com/cocodataset/cocoapi.git follow above medium for how
Download protobuf follow above medium for how
Extract and Execute protobouf
From conda environment set PYTHONPATH=\tensorflow1\models;\tensorflow1\models\research;\tensorflow1\models\research\slim
Move track.py and ipcam_utils.py to the models/research/object_detection folder
From the object detection folder:
Run the track.py "python tracker_test.py --Camera_IP  --Camera_User  --Camera_PW "

ipcam_utils.py

https://s3.amazonaws.com/fdt-files/FDT+IP+Camera+CGI+%26+RTSP+User+Guide+v1.0.2.pdf

Track.py

https://www.edureka.co/blog/tensorflow-object-detection-tutorial/
