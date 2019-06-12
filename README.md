# IP Camera

These scripts are used to send HTTP commands to an IP Camera. These commands will focus an object (using Tensorflow Object Detection repo) to place a subject in the center of a frame. Place script files in the object detection directory. 

#### V2 

The threaded version of the camera tracker run's a lot more efficient in real-time. Object detection predictions take about 0.3s to run. 

- [ ] Thread each process in an efficient manner
- [x] Thread object detection
- [x] Thread person tracker
- [x] Thread videa stream/frames

#### Requires:

- Tensorflow 1.12.0
- Anaconda
- OpenCV2

#### Installation Instructions:

These were the tutorial's and instruction I followed to get the object detector to work:

- [Object Detection API Tutorial for windows](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
- [Tensorflow Object Detection API windows](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b)

1. Clone the model repo from tensorflow https://github.com/tensorflow/models

2. Conda create -n "env name" python=X.X anaconda [Conda Environment Creation](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

3. Conda install tensorflow
4. pip install other dependencies listed in the above
5. Install coco api by `git clone https://github.com/cocodataset/cocoapi.git` follow the above medium article on how
6. Download protobuf follow above medium for how
7. Extract and Execute protobouf
8. From the conda environment `set PYTHONPATH=\tensorflow1\models;\tensorflow1\models\research;\tensorflow1\models\research\slim`
9. Move `track.py` and `ipcam_utils.py` to the `models/research/object_detection` folder
10. From the object detection folder:
  - Run the track.py "python tracker_test.py --Camera_IP  --Camera_User  --Camera_PW "

**ipcam_utils.py**

This file follows the manufacturers (Dericam) guidelines for making requests to the camera.

- [Dericam User Guide](https://s3.amazonaws.com/fdt-files/FDT+IP+Camera+CGI+%26+RTSP+User+Guide+v1.0.2.pdf)

**Track.py**

Used the below tutorial to have real-time object detection (unsurprisingly this is slow compared to the threaded version).

- [Object detection tutorial](https://www.edureka.co/blog/tensorflow-object-detection-tutorial/)
