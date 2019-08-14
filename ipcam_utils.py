import requests
import argparse
import cv2
import sys
import re
import time
import random
from random import randint

"""
    Dericam Manual:
    https://s3.amazonaws.com/fdt-files/FDT+IP+Camera+CGI+%26+RTSP+User+Guide+v1.0.2.pdf 

    All cgi commands need authentication before they can be executed. 
    There are 2 kinds of authentications mechanism. The first
    one is the Basic authentication mechanism declared by HTTP protocol. The second
    one is including the username/password in the paramaters of CGI command URL.
    We only support GET and POST method of HTTP CGI. The command upgrade.cgi
    and restore.cgi are used for the POST method, all others commands are GET
    methods.

    CGI can be executed in the URL of web bowser, which are as follows:
        http://192.168.1.88/cgi-bin/hi3510/param.cgi?cmd=getvencattr&-chn=11
        http://192.168.1.6/cgi-bin/hi3510/param.cgi?cmd=setvencattr&-chn=11&-fps=15
        http://192.168.1.6/cgi-bin/hi3510/param.cgi?cmd=getwirelessattr
     Or you can include the username/password of the IP camera directly:
        http://192.168.1.88/cgi-bin/hi3510/param.cgi?cmd=getvencattr&-chn=11&-usr=admin&-pwd=admin
        http://192.168.1.6/cgi-bin/hi3510/param.cgi?cmd=getwirelessattr&-usr=admin&-pwd=admin
"""

#############################################################################
#PAN-TILT-ZOOM

class PTZ_commands(object):
    """
    A PTZ object that sends http requests to an Dericam, IP Camera. 
        http://10.32.100.58/cgi-bin/hi3510/param.cgi?cmd=ptzctrl&-step=0&-act=left&-speed=45
        https://www.manualslib.com/manual/1432172/Dericam-Cgi.html?page=31#manual
        usage: /cgi-bin/hi3510/param.cgi?cmd=ptzctrl.cgi[&-step=&-act=&-speed=]
        Example: /cgi-bin/hi3510/param.cgi?cmd=ptzctrl&-step=0&-act=left&-speed=45
        Param:
            Step : 
                0 = single step and stop until stop command
                1 = single step and stop automatically
            act - command strings for motion control :
                left: Go left
                right: Go right
                up: Go up
                Down: Go down
                upleft, upright, downleft, downright: Go to upleft,upright, downleft, downright
                home: Back to center point.
                zoomin: Zoom in.
                zoomout: Zoom out.
                hscan: The curise of horizontal.
                vscan: The curise of vertical.
                stop: Stop.
            speed : value 1-63
        Returns success or fail
    """
    def __init__(self, user, pwd, IP, port=554):
        self.ip = IP
        self.base_url = "http://{}/cgi-bin/hi3510/param.cgi".format(self.ip)
        self.ptzctrl = "?cmd=ptzctrl"
        self.step = "&-step="
        self.speed = "&-speed="

        self.Commands = {}
        self.Commands['LEFT'] = "&-act=left"
        self.Commands['RIGHT'] = "&-act=right"
        self.Commands['UP'] = "&-act=up"
        self.Commands['DOWN'] = "&-act=down"
        self.Commands['STOP'] = "&-act=stop"
        self.Commands['ZOOMIN'] = "&-act=zoomin"
        self.Commands['ZOOMOUT'] = "&-act=zoomout"
        self.Commands['HOME'] = "&-act=home"
        self.Commands['HSCAN'] = "&-act=hscan"
        self.Commands['VSCAN'] = "&-act=vscan"
        self.Commands['SPEED'] = "&-speed="
        self.Commands['STEP'] = "&-step="
        self.Commands['GETMDATTR'] = "?cmd=getmdattr"
        self.Commands['SETMDATTR'] = "?cmd=setmdattr"
        self.Commands['ENABLE'] = "&-enable="
        self.Commands['SENSITIVITY'] = "&-s="
        self.Commands['NAME'] = "&-name="
        self.Commands['X'] = "&-x="
        self.Commands['Y'] = "&-y="
        self.Commands['W'] = "&-w="
        self.Commands['H'] = "&-h="

        self.Commands['GETMOTOATTR'] = "?cmd=getmotorattr"
        self.Commands['GETMDATTR'] = "?cmd=getmdattr"

        self.user = user
        self.pwd = pwd 
        self.port = port
        self.rtsp = "rtsp://%s:%s@%s:%s/11" % (self.user, self.pwd, self.ip, self.port)
        self.auth = "&-usr={}&-pwd={}".format(user, pwd)

    def pan(self, Direction, Steps, Speed):
        Speed = str(Speed)
        Steps = str(Steps)
        direction = Direction.upper()
        comm = self.base_url + self.ptzctrl + self.Commands['STEP'] + Steps + self.Commands[direction] +\
         self.speed + Speed + self.auth
        res = requests.get(comm)
        print(direction, res.status_code)
        return res

    def setmottoattr(self, panspeed, tiltspeed, 
        panscan, tiltscan, movehome, ptzalarmmask, alarmpresetindex):
        """
        We can use the motion detector to track objects by passing in the coordinates as a parameter.
        We then track these attributes maybe in an object in the constructor or via the method getmdattr()
        Then the complicated bit: 
            based on the velocity and trajectory of an object (within some criteria) set the ptz controller to pan
            and to zoom 

        panspeed: Speed of pan, 0 - Fast, 1 - Medium, 2 - Slow
        tiltspeed: Speed of tilt, 0 - Fast, 1 - Medium, 2 - Slow
        panscan: number for horizontal cruise
        tiltscan: the number for vertical cruise
        movehome: back to center after self-check, on - Enabled, off - Disabled
        ptzalarmmask: Close the motion alarm while PTZ is moving.
        alarmpresetindex: Alarm link action for going to preset, the preset number if 1-8
        """
        comm = self.base_url + self.Commands['GETMOTOATTR'] + str(panspeed) + str(tiltspeed) +\
         str(panscan) + str(tiltscan) + movehome + ptzalarmmask + str(alarmpresetindex) + self.auth
        res = requests.get(comm)
        print(res.status_code)
        return res

    def left(self):
        #Basic Left Commands
        comm = "http://{}/cgi-bin/hi3510/ptzleft.cgi".format(self.ip)# + self.auth
        res = requests.get(comm)
        print(res.status_code)
        return res

    def right(self):
        #Basic Right Commands
        comm = "http://{}/cgi-bin/hi3510/ptzright.cgi".format(self.ip)# + self.auth
        res = requests.get(comm)
        print(res.status_code)
        return res

    def getmdattr(self):
        """
        Get the information from of coordinates for motion detector. 
        """
        comm = self.base_url + self.Commands['GETMDATTR'] + self.auth
        res = requests.get(comm)
        print(res.status_code)
        return comm, res

    def getmottoattr(self):
        comm = self.base_url + self.Commands['GETMOTOATTR'] + self.auth
        res = requests.get(comm)
        print(res.status_code, '\n', res.text)
        return res

    def getMotion(self):
        comm = self.base_url + self.Commands['GETMDATTR'] + self.auth
        res = requests.get(comm)
        print(res.status_code, '\n', res.text)
        return res

    def setMotion(self, enable, s, window, x, y, w, h):
        """
        get x, y, w, h co-ordinates from object detector.
        usage: /cgi-bin/hi3510/param.cgi?cmd=setmdattr&-enable=1&-s=50&-name=1&-x=0&-y=0&-w=60&-h=60
        """
        comm = self.base_url + self.Commands['SETMDATTR'] + self.Commands['ENABLE'] + str(enable) +\
         self.Commands['SENSITIVITY'] + str(s) + self.Commands['NAME'] + str(window) + self.Commands['X'] + str(x) +\
         self.Commands['Y'] + str(y) + self.Commands['W'] + str(w) + self.Commands['H'] + str(h) + self.auth
        res = requests.get(comm)
        print(res.status_code)
        return res

    def tilt(self):
        return

    def zoom(self, zoom, Steps, Speed):
        Speed = str(Speed)
        Steps = str(Steps)
        comm = self.base_url + self.ptzctrl + self.step + Steps + self.Commands[zoom] + self.speed + Speed + self.auth
        res = requests.get(comm)
        print(res.status_code)
        return res

#############################################################################
#Not Used

class Camera(object):

    def __init__(self, PTZ_Controller):
        self.cam_height = 0
        self.cam_width = 0
        self.ptz_controller = PTZ_Controller

        self.Coordinates = {
            'left' : 0,
            'right' : 0,
            'top' : 0,
            'bottom' : 0
        }
        self.motion_enabled = 0
        self.coord_update = False
        self.Frame = 0
        self.imshowIMG = 0
        #self.video = cv2.VideoCapture(self.ptz_controller.rtsp)
        self.cam_width = 0 #self.video.get(cv2.CAP_PROP_FRAME_WIDTH )
        self.cam_height = 0 #self.video.get(cv2.CAP_PROP_FRAME_HEIGHT )

    def startStream(self):
        """
        start stream via rtsp
        rtsp://<user>:<pw>@<ip>:<port>/11
        """
        self.video
        cv2.imshow('object detection', cv2.resize(self.imshowIMG, (800,600)))

    def CamRead(self):
        ret, self.Frame = self.video.read()
        return

    def getCoordinateInfo(self):
        """
        Get the information from of coordinates for motion detector. 
        """
        _, attr = self.ptz_controller.getmdattr()
        parse = "".join(attr.text)
        parse = parse[0:135]
        matches=re.findall(r'\"(.+?)\"',parse)
        dictionary = {}
        arr = ['enable', 'x', 'y', 'w', 'h', 'sensitivity', 'threshold']
        for i, val in enumerate(matches):
            dictionary[arr[i]] = val
        self.updateCoordinates(dictionary)
        print(parse)
        self.coord_update = True
        return parse

    def updateCoordinates(self, newValues):
        self.Coordinates = newValues
        return

    def drawRectangle(self, frame):
        x1 = int(self.Coordinates['left'])
        y1 = int(self.Coordinates['right'])
        y2 = int(self.Coordinates['top'])
        x2 = int(self.Coordinates['bottom'])
        return cv2.rectangle(frame, (x1,y1),(x2,y2),(255, 0, 0), 3)

    def CenterObject(self, left, right, top, bottom):
        """
        This assumes width = 1920, height = 1080.
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
        updateCoordinates((left, right, top, bottom))
        centered = True
        inv_left = left - (1920 * 0.1)
        inv_right = right - (1920 * 0.9)
        inv_top = top - (1080 * 0.9)
        has_moved = False

        if right > (1920 * 0.9):  
            print("Going right : \nright: ",right, (1920 * 0.9), '******\n')
            self.ptz.pan(Direction='right', Steps=1, Speed=1)
            time.sleep(10)
            has_moved = True
            print("Movement has finished.")
            centered = False

        if left < (1920 * 0.1):
            print("Going left : \nleft: ", left, (1920 * 0.1), '******\n')
            self.ptz.pan(Direction='left', Steps=1, Speed=1)
            time.sleep(10)
            has_moved = True
            print("Movement has finished.")
            centered = False

        if top > (1080 * 0.9):
            print("Going Down : \ntop: ", top, (1080 * 0.9), '\n******\n')
            self.ptz.pan(Direction='down', Steps=1, Speed=1)
            time.sleep(10)
            has_moved = True
            print("Movement has finished.")
            centered = False

        if centered == True:
            print("The person is in frame.\n")
            print("The object is : ", round(inv_left), " from the left before we move.\n")
            print("The object is : ", round(inv_right), " from the right before we move.\n")
            print("The object is : ", round(inv_top), " from the top before we move.\n")

        return has_moved

    def findObject(self):
        has_completed_once = False
        delay_time = 10
        random_move = randint(1,6)
        delay_time = 10

        if random_move == 1:
            print("Going right\n")
            self.ptz.pan(Direction='right', Steps=1, Speed=1)
            time.sleep(delay_time)
            has_completed_once = True
            print("Movement has finished.")

        if random_move == 2:
            print("Going left\n")
            self.ptz.pan(Direction='left', Steps=1, Speed=1)
            time.sleep(delay_time)
            has_completed_once = True
            print("Movement has finished.")

        if random_move == 3:
            print("Going Down\n")
            self.ptz.pan(Direction='down', Steps=1, Speed=1)
            time.sleep(delay_time)
            has_completed_once = True
            print("Movement has finished.")

        if random_move == 4:
            print("Going Up\n")
            self.ptz.pan(Direction='up', Steps=1, Speed=1)
            time.sleep(delay_time)
            has_completed_once = True
            print("Movement has finished.")

        if random_move == 5:
            print("Horizontal Scan\n")
            self.ptz.pan(Direction='hscan', Steps=1, Speed=1)
            time.sleep(delay_time)
            has_completed_once = True
            print("Movement has finished.")

        if random_move == 6:
            print("Vertical Scan\n")
            self.ptz.pan(Direction='vscan', Steps=1, Speed=1)
            time.sleep(delay_time)
            has_completed_once = True
            print("Movement has finished.")

        return has_completed_once     
