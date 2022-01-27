# -*- coding: utf-8 -*

# Laser pointer detector, V3.
# ---------------------------

# Copyright (C) 2018-2022 San Vu Ngoc
# Université de Rennes 1

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# This version uses python3 and openCV >= 3

# THANK YOU stackoverflow, opencv docs, https://www.pyimagesearch.com, etc.
# TODO conform to style:
# https://www.python.org/dev/peps/pep-0008/

import argparse
import tempfile
import os.path
from random import randint
import timeit
import math

import yaml
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Camera constants that are missing in some versions of OpenCV's python bindings
try:
    CAM_CONTRAST = cv2.CAP_PROP_CONTRAST
    CAM_GAIN = cv2.CAP_PROP_GAIN
    CAM_BRIGHTNESS = cv2.CAP_PROP_BRIGHTNESS
except NameError:
    CAM_CONTRAST = 11
    CAM_GAIN = 14
    CAM_BRIGHTNESS = 10

# BGR colors
MAXVAL_COLOR = (0,255,230) # yellow
PREDICTED_COLOR = (155,45,0) # dark blue
VAL_COLOR = (45,23,240) # bright red
CROP_COLOR = (12,234,54) # green
SNAKE_COLOR = (5,200,160) # yellow-green
BUTTONDOWN_COLOR = (90,90,160) # orange

# alternative way of capturing webcam:
# ffmpeg -y -f v4l2 -i /dev/video0 -update 1 -r 25 output.bmp
USE_FFMPEG = False

# Global debugging variable (can be modified)
gdebug = True

# Remark: NO tuple pattern matching in python 3 anymore... :(
# https://www.python.org/dev/peps/pep-3113/
def color_plt(color):
    """convert openCV color to mathplotlib color"""
    b,g,r = color
    return (np.array((r/255.0, g/255.0, b/255.0)))
    
# Some debugging utilities
def printd(s):
    global gdebug
    if gdebug:
        print(s)

def print_time(s, t0, thr=0.0001):
    global gdebug
    t = timeit.default_timer() - t0
    if gdebug and t >= thr:
        print ("TIME %s = %f"%(s,t))
    return (t)

def view_color(c): # not used
    cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
    image = np.zeros((100,100,3), np.uint8)
    image[:,:] = c
    cv2.imshow("Color", image)
    _ = cv2.waitKey(0)
    cv2.destroyWindow("Color")


# At the heart of the algorithm we use background signed substraction for a
# fast "motion" detector.
#
# Warning, the resulting diff is not a valid openCV image, it can contain
# negative integers
def diff_max(img1, img2, radius=2):
    """signed difference img2 - img1 and max value and position"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # We blur the images by radius at least 3. For 640x480 webcam, 3 seems to
    # substancially improve laser detection.
    radius = 2 * int((max(2, radius)) // 2) + 1 # should be an odd number
    gray1 = cv2.GaussianBlur(gray1, (radius, radius), 0)
    gray2 = cv2.GaussianBlur(gray2, (radius, radius), 0)
    h,w = gray1.shape
    diff = gray2.astype(int) - gray1.astype(int)
    imax = np.argmax(diff)
    x, y = (imax % w, imax // w)
    maxVal = diff[y,x]
    maxLoc = (x,y)
    return (diff, maxVal, maxLoc)
        
def get_angle(p1,p2):
    """angle in rad between 2 vectors (shape (2,)"""
    l1 = np.linalg.norm(p1)
    l2 = np.linalg.norm(p2)
    if l1 == 0. or l2 == 0.:
        return 0
    else:
        d = p1.dot(p2)/(l1*l2)
        if d > 1.:
            d = 1
        elif d < -1.:
            d = -1
        a = math.acos(d)
        det = np.linalg.det([p1, p2])
        if det >= 0:
            return a
        else:
            return -a

class Console:
    """Rudimentary image+text console

    Used to display result of detection and instructions"""
    def __init__(self, name, imgsize, textheight):
        w,h = imgsize
        self.size = imgsize # this should not be changed
        self.name = name
        self.topmargin = 15
        self.leftmargin = 10
        self.hline = self.topmargin
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_scale = 1.5
        self.thickness = 2
        self.color = (5,15,67)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w, h + textheight)
        self.image = np.zeros((h, w, 3), np.uint8)
        self.cons =  np.ones((textheight, w, 3), np.uint8) * 255
        self.total = np.zeros((h + textheight, w, 3), np.uint8)
        
    def show(self):
        ih,iw,_ = self.image.shape
        th,tw,_ = self.cons.shape
        if iw != tw:
            print ("Error text width " + str(tw) +" and image width " + str(iw) + "should be equal")
        self.total[0:ih, :] = self.image
        self.total[ih : ih+th, :] = self.cons
        cv2.imshow(self.name, self.total)

    def force_show(self):
        self.show()
        _ = cv2.waitKey(1)

    def set_image(self, image):
        # we don't want to change the self.size so we need to resize image
        # before displaying if it doesn't have to right shape. Warning, this
        # slows it down...
        h,w,_ = image.shape
        if (w,h) != self.size:
            print ("RESIZING " + str((w,h)) + " to " + str(self.size))
            self.image = cv2.resize(image, self.size)
        else:
            self.image = image
            
    def show_image(self, image):
        self.set_image(image)
        self.show()

    def wait_key(_, n):
        return(cv2.waitKey(n))
               
    def write(self, text):
        """Write one line of text to the console and advance line"""
        (tw,th), baseVal = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        w,h = self.size
        if tw + self.leftmargin > w:
            scale = self.font_scale*w/float(tw + self.leftmargin)
        else:
            scale = self.font_scale
        printd ("[CONSOLE]: " + text)
        cv2.putText(self.cons, text, (self.leftmargin, self.hline + th),
                    self.font, scale, self.color, self.thickness,
                    cv2.LINE_AA, False)
        self.hline = self.hline + th + baseVal

    def reset(self):
        """Clear text area (but leaves the image area)"""
        self.cons[:,:,:] = 255
        self.hline = self.topmargin

    def close(self):
        cv2.destroyWindow(self.name)

class Snake:
    """Class for snake operations (and mouse. TODO move mouse to another class?)

    The 'snake' is the list of last contiguously detected pixels,
    like in the classic 'snake' game.
    """

    def __init__(self, snakeMaxSize):
        self.maxSize = snakeMaxSize
        self.points = np.zeros((snakeMaxSize,2,2), dtype=int)
        # we start from the end of the array... don't ask why.
        # points[i][0] = point with index i
        # points[i][1][0] = frame number when point was added
        self.size = 0
        self.last_active_frame = 0
        self.active = False # is the light on?
        self.current_frame = 0
        self.max_age = 3 # OK ?
        self.target = None # mouse click position
        self.target_frame = 0
        self.target_radius = 30 # mettre cal.jitterDist/2
        self.button_down = False
        self.click = False

    def next_frame(self):
        self.current_frame += 1
        if self.age() > self.max_age: # too old, we kill it to start from stratch
            self.size = 0
            self.button_down = False
            self.click = False
            
    def frame(_, point):
        return (point[1][0])

    # number of frames elapsed since last point was added
    def age(self):
        if self.size == 0:
            return (-1)
        else:
            return (self.current_frame - self.frame(self.points[-1]))
        
    def visible(self):
        """return the visible part"""
        start = len(self.points) - self.size
        return (self.points[start:,0,:])

    def empty(self):
        return (self.size == 0)
    
    def last(self):
        """return last point"""
        if self.size == 0:
            print ("ERROR: snake empty")
        return (self.points[-1][0])

    def lastn(self,n):
        """return last nth point"""
        if self.size == 0:
            print ("ERROR: snake empty")
        return (self.points[-n][0])

    def length(self): # not used yet
        """mean length of a segment, in pixels"""
        # about 1e-5 sec
        if self.size == 0:
            return (0.0)
        else:
            return (cv2.arcLength(self.visible(), False) / self.size)

    def area(self, image): # not used yet
        """Area of best rectangular approximation. Area = 1 <=> whole image"""
        # about 5e-05 sec, incl drawing
        box = cv2.minAreaRect(self.visible())
        pts = np.array(cv2.boxPoints(box), dtype=int)
        # draw the result on the image
        cv2.polylines(image, [pts], True, (0,0,200), 1)
        h,w,_ = image.shape
        return (cv2.contourArea(pts) / (h*w)) #plus rapide: utiliser box!

    def enclosing_circle(self):
        """center and radius of best enclosing circle"""
        return(cv2.minEnclosingCircle(self.visible()))  # → center, radius
        
    def _target_locked(self, radius):
        # are we pointing at the same spot for long enough ?
        min_size = 8
        # snake has to be at least this size to be considered locked. TODO use
        # FPS/4 or 2*maxSize/3...
        if self.size >= min_size:
            center, r = self.enclosing_circle()
            if r <= radius:
                return (True, (int(center[0]), int(center[1])))
            else:
                return (False, None)
        else:
            return (False, None)
               
    def predict(self):
        """predict the next point by 2nd order curvature approx"""
        # about 1e-04 sec, why so "slow"?
        snake = self.points
        if self.size == 0:
            # cannot predict
            return (np.array([0,0], dtype=int))
        elif self.size == 1:
            return (snake[-1][0])
        else:
            dt = self.current_frame - self.frame(self.points[-1])
            # dt = same as self.age(), but don't need to test size here.
            printd ("dt=" + str(dt))
            if dt != 1:
                printd ("** A point was probably lost")
                dt = min(3, dt)
                # it doesn't make sense to predict very far if we lost
                # more than a couple of frames.
            # Si on appelle predict seulement quand on est 'active', on aura
            # toujours dt = 1. Sinon ça peut permettre de 'rattraper' un point
            # perdu (défaut de lumière ponctuel, par exemple).
            if self.size == 2:
                # first order
                l = snake[-1] - snake[-2]
                return ((snake[-1] + dt*l)[0])
            else: #second order
                l1 = snake[-2] - snake[-3]
                l2 = snake[-1] - snake[-2]
                a = get_angle(l2[0], l1[0]) * 180 / math.pi
                rot = cv2.getRotationMatrix2D((0,0), dt * a, 1) # shape (2,3)
                r = np.matmul(rot,np.array([l2[0][0], l2[0][1], 0]))
                return (snake[-1][0] + dt*r.astype(int))

    def grow(self, point):
        """add a point to the snake"""
        # we add to its end, shifting the others to the left:
        snake = self.points
        l = len(snake)
        snake[0:l-1] = snake[1:l]
        snake[l-1][0] = point
        snake[l-1][1] = [self.current_frame,0]
        self.last_active_frame = self.current_frame
        if self.size == l:
            s = self.size
        else:
            s = self.size + 1
        self.size = s
        if self.size == 1:
            print ("(tentative)"),
        print ("point #" + str(self.current_frame) + " = " + str(point))
        # WARNING: for more safety, this point should be considered valid only
        # if snake size >= 2 (if one can afford to wait for another frame).
        if (not self.button_down) and self.target is not None:
            d = np.linalg.norm([self.target[0],self.target[1]] - point)
            age = self.current_frame - self.target_frame
            print ("TARGET dist = %f, age = %d"%(d, age))
            if age > 25: # TODO use FPS instead of 25
                # then we forget about the target
                self.target = None
                self.click = False
            elif self.size >= 2:
                if d < self.target_radius:
                    print ("*** MOUSE BUTTON DOWN ***")
                    self.button_down = True
                else: # forget the target
                    self.target = None
                    self.click = False
        
    def remove(self):
        """remove the first point = tail of the snake"""
        if self.size == 0:
            print ("ERROR: cannot remove point of empty snake")
        else:
            # before removing point we check "mouse" events: if the "target
            # should be locked" or if click is validated.  En principe si
            # self.button_down alors target <> None, mais bon, on teste quand
            # même...
            if (self.button_down and self.target is not None and
                    np.linalg.norm([self.target[0],self.target[1]] - self.last()) < self.target_radius):
                print ("-----------CLICK! " + str(self.target) +  " ----------")
                self.click = True
                self.button_down = False
            elif self.target is None:
                locked, target = self._target_locked(self.target_radius)
                if locked:
                    self.target = target
                    self.target_frame = self.current_frame
            self.size = self.size - 1

    def draw(self, image):
        """draw the snake on the image"""
        color = (BUTTONDOWN_COLOR if self.button_down else SNAKE_COLOR)
        if (self.size == 1 and
                (self.frame(self.points[-1]) == self.current_frame)):
            # New point starting a new snake. This point should not be
            # considered 100% valid if it stays isolated.
            p = self.last()
            cv2.circle(image, (p[0], p[1]), 5, PREDICTED_COLOR, -1, cv2.LINE_AA)
            
        elif self.size >= 1:
            cv2.polylines(image, [self.visible()], False, color, 2,
                          cv2.LINE_AA)
            if self.active:
                p = self.last()
                cv2.circle(image, (p[0], p[1]), 5, color, -1, cv2.LINE_AA)
        if self.target is not None:
            thick = (-1 if self.click else 2)
            cv2.circle(image, self.target, self.target_radius,
                       BUTTONDOWN_COLOR, thick, cv2.LINE_AA)

class Background:
    """Class for background accumulation"""
    
    def __init__(self, length):
        self.length = length # max number of images
        self.empty = True
        self.images = [] # list of images
        self.sum = [] # sum of all images
        if gdebug:
            cv2.namedWindow("BACKGROUND", cv2.WINDOW_NORMAL)

    def add(self, image):
        l = len(self.images)
        self.images.append(image)
        if self.empty:
            self.empty = False
            m = image.astype('int')
        else:
            m = self.sum + image.astype('int')
        if l >= self.length:
            # we discard the element #0 from the images (and remove it from the
            # sum)
            m = m - self.images[0]
            self.images = self.images[1:]
        self.sum = m

    def mean(self):
        if self.empty:
            print ("Error: background empty")
            return (None)
        else:
            b = (self.sum/len(self.images)).astype(np.uint8)
            if gdebug:
                cv2.imshow("BACKGROUND", b)
            return (b)

    def close_window(_):
        cv2.destroyWindow("BACKGROUND")
        
# Here we detect "global motion", which is when we think the change in the
# image is too important to be due to the laser pointer.
#
def globalMotion(gray, threshold):
    """percentage in [0,1] of moved pixels"""
    # This algorithm is efficient when there is little motion, which is the
    # case in principle most of the time.
    #
    # 0.000334 sec for globalMotion = 0.5
    # 0.000104 sec for globalMotion = 6e-06
    moved = gray[gray>threshold]
    h,w = gray.shape
    # pour obtenir leur position (linearisée): np.nonzero(gray > 10))
    return (len(moved) / float (w*h))


# find connected component of initial point maxLoc of pixels with gray value >=
# threshold.  If we switch to opencv3, we could use cv2.connectedcomponents
def laserShape(diff, maxLoc, threshold, maxRadius=100, debug=True):

    printd("--------Laser shape---------(maxRadius=" + str(maxRadius) + ")---")
    x,y = maxLoc
    lowDiff = max(0, diff[y,x] - threshold)
    printd ("maxLoc = " + str(maxLoc) + ", selecting from " + str(lowDiff) + " to " + str(diff[y,x]))
    left, top = x - maxRadius, y - maxRadius # can be negative
    gh,gw = diff.shape
    right, bottom = min(x + maxRadius + 1, gw), min(y + maxRadius + 1, gh)
    crop2 = diff[max(0,top) : bottom, max(0,left) : right]
    crop = np.absolute(crop2).astype(np.uint8) # we convert to a format that is a valid opencv image
    h,w = crop.shape # in principle each size= 2*maxRadius, but crop2 may be
                     # chopped if maxLoc is close to the boundary of the
                     # image
    mask = np.zeros((h+2, w+2), np.uint8) # we add a 1 pixel border
    printd ("Mask shape = " + str(mask.shape))
    printd ("Crop shape = " + str(crop.shape))
    printd ("Diff shape = " + str(diff.shape))

    # Filling with white. The point 'seed' should correspond to (x,y) in the
    # original diff image. The value "newVal=125" is ignored because of the
    # flag FLOODFILL_MASK_ONLY.
    seed = maxRadius + min(0, left), maxRadius +  min(0, top)
    printd ("Seed = " + str(seed))
    retval, _, mask, rect = cv2.floodFill(crop, mask, seed, 125, lowDiff, 255, cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | 4 | ( 255 << 8 ) )
    if debug:
        print ("floodFill retval (#of filled pixels) = " + str(retval))
        (rx,ry,rw,rh) = rect
        cv2.rectangle(crop, (rx,ry), (rx+rw-1,ry+rh-1), (125,125,125))
        cv2.namedWindow("Diff", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Crop", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Diff", np.absolute(diff).astype(np.uint8))
        cv2.imshow("Mask", mask)
        cv2.circle(crop, seed, 5, (150,22,56), 1)
        cv2.imshow("Crop", crop)
        print ("Found bounding box = " + str(rect))
        print ("Press any key")
        _ = cv2.waitKey(100) # 0
        
    # find rotated box TODO USE THIS...
    (rx,ry,rw,rh) = rect
    clip = mask[ry:ry+rh+2,rx:rx+rw+2]
    # Warning, the function findContours modifies the image,
    # so we make a copy.
    #pts, _ = cv2.findContours(clip.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pts = np.argwhere(np.transpose(clip)) # extract indices (x,y) where clip entries are not zero
    #line = cv2.fitLine(pts[0], cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
    angle = 0.
    drift = 0
    if len(pts) > 0:
        center, dims, angle = cv2.minAreaRect(pts.reshape(len(pts),1,2))
        drift = int(abs(dims[1] - dims[0]))
        if dims[0] < dims[1]:
            mainAngle = angle + 90.0
        else:
            mainAngle = angle
        printd ("Angle = " + str(mainAngle) + ", Drift = " + str(drift))
        if debug:
            box = np.array(cv2.boxPoints((center, dims, angle)), dtype=int)
            printd ("Box points ) " + str(box))
            colorclip = cv2.cvtColor(clip, cv2.COLOR_GRAY2BGR)
            cv2.polylines(colorclip, [box], True, (255,123,55), 1)
            r = max(dims)/2
            cx, cy = int(center[0]), int(center[1])
            x1 = cx + int(r * math.cos(math.pi * mainAngle/180.))
            y1 = cy + int(r * math.sin(math.pi * mainAngle/180.))
            cv2.line(colorclip, (cx, cy), (x1, y1), (24,123,55), 1)
            cv2.imshow("Mask", colorclip)
            print ("Press any key")
            _ = cv2.waitKey(100) # 0
            
    else:
        printd ("ERROR: Cannot find laser pixels!")
        
    return (mask, rect, angle)

# Return a list of (nProbes-1) pairs of images [background, image], and the
# list of maxVals
def getProbes(cam, nProbes, clipBox, console, drawFn=None, bkgLen=10):
    # cf webcamTracker for explanation of the variables
    vals = []
    images = []
    background = Background(bkgLen)
    cal = Calibration(cam)
    snake = Snake(nProbes-1)
    printd ("Please wait...")
    t0 = timeit.default_timer()
    for i in range(nProbes):
        img = readCam(cam)
        snake.next_frame()
        if i != 0: # ie. if background not empty we store the mean image
            images.append([background.mean(), img])
        show = img.copy()
        if drawFn is not None:
            drawFn(show)
        _ = cv2.waitKey(1)
        mask, maxVal = oneStepTracker(background, img, show, clipBox, snake, cal)
        if i != 0:
            vals.append(maxVal)
        console.show_image(show)
    printd ("Avg FPS for " + str(nProbes) + " cam probes =" + str(nProbes/(timeit.default_timer()-t0)))
    return (vals, images)

def camSize(cam):
    return (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def openCam(camera_id, cam_api=cv2.CAP_V4L):
    cam = cv2.VideoCapture(camera_id, cam_api)
    if not cam.isOpened():
        print ("Could not open Camera #" + str(camera_id))
        print ("Trying another one...")
        cam = cv2.VideoCapture(-1)
        if not cam.isOpened():
            if cam_api == cv.CAP_ANY:
                print ("All webcam failed. Quitting.")
                exit()
            else:
                openCam(camera_id, cam_api=cv.CAP_ANY)
    width, height = camSize(cam)
    print ("Cam opened with size = " + str(width) + ", " + str(height))
    return (cam, width, height)

def readCam(cam):
    "Read image from webcam."
    # Warning: this is the slowest part of all. Depending on camera, we
    # typically have 30 FPS or 20 FPS. ffmpeg is faster.
    if USE_FFMPEG:
        essai = 0
        retval = False
        while not retval and essai < 2:
            img = cv2.imread("/tmp/output.bmp")
            retval = (img is not None)
            essai += 1
            if not retval:
                cv2.waitKey(5)
    else:
        retval, img = cam.read()
    if not retval:
        print ("Cannot capture frame device")
        img = cv2.imread("webcam_error.jpeg")
    return (img)

# Class to store calibration values
class Calibration:
    def __init__(self, cam):
        w, h = camSize(cam)
        self.width, self.height = w, h
        self.diag = math.sqrt(w*w + h*h)
        # don't change brightness and contrast during calibration!
        self.brightness = cam.get(CAM_BRIGHTNESS)
        self.contrast= cam.get(CAM_CONTRAST)
        # default values:
        self.motionThreshold = 7
        # = difference of intensity above which we consider moved pixels.
        self.globalMotionThreshold = 0.001
        # = percentage of moved pixels above which we decide that a global
        # motion has occured (and thus the image should not be analyzed).  It
        # depends on pointer size and camera resolution.
        self.laserIntensity = 40
        # Reaching this intensity qualifies for being the laser beam.
        self.laserDiameter = 5 # size of laser dot in pixels.
        self.jitterArea = 0.003 # jitter area due to shaky hand; à détecter
        self.jitterDist = math.sqrt(self.jitterArea * h * w) # here 30

    def save(self, filename): # uses yaml module
        """Save calibration data to file"""
        print ("Saving calibration to " + filename)
        with open(filename, 'w') as outfile:
            yaml.dump(self, outfile)

    def load(self, cam, filename):
        """Update calibration data by loading file"""
        print ("Loading calibration from " + filename)
        with open(filename, 'r') as stream:
            try:
                x = yaml.load(stream)
                # for safety we select the attributes manually:
                for tag in ['width', 'height', 'brightness', 'contrast',
                            'diag', 'motionThreshold',
                            'globalMotionThreshold', 'laserIntensity',
                            'laserDiameter', 'jitterArea', 'jitterDist' ]:
                    setattr(self, tag, getattr(x, tag))
            except yaml.YAMLError as exc:
                print(exc)
        cam.set(CAM_BRIGHTNESS, self.brightness)
        cam.set(CAM_CONTRAST, self.contrast)
            
# We run another pass on a recorded set of images. At this point this requires
# user input. I'm thinking of making this more automatic.
def optimize(cal, images, clipBox, console):
    global gdebug
    
    p1,p2 = clipBox
    w,h = cal.width, cal.height
#    consoleHeight = 60
#    console = np.zeros((consoleHeight, 2*w, 3), np.uint8)
    
    snake = Snake(len(images))
    background = Background(1)
    print ("Please wait...")
    for pair in images: # pair = [bkg, img]
        show = pair[1].copy()
        snake.next_frame()
        cv2.rectangle(show, p1, p2, CROP_COLOR)
        p = snake.predict()
        background.add(pair[0])
        mask, maxVal = oneStepTracker(background, pair[1], show, clipBox, snake, cal)
        if snake.active:  # draw the cross at the chosen point
            [x,y] = snake.last()
            cv2.line(show, (0,y), (w,y), MAXVAL_COLOR, 1)
            cv2.line(show, (x,0), (x,h), MAXVAL_COLOR, 1)
        
        # we recompute the candidates (in order to optimize, we should use the
        # ones computed in oneStepTracker...)
        diff, maxVal, maxLoc = diff_max(pair[0], pair[1], cal.laserDiameter/2)
        cc = maxValPos(diff, cal.motionThreshold, 10)
        for c in cc:
            printd (c)
            plotVal(show, [c[0], c[1], c[2]*3])
            # we multiply intensiy by 3 to make it more visible

        console.reset()
        console.write("motionThreshold=" + str(cal.motionThreshold))
        console.write("snake size = " + str(snake.size))
        console.write("active = " + str(snake.active))

        dual = np.zeros((h, 2*w, 3), np.uint8)
        dual[0:h,0:w] = pair[1]
        dual[0:h,w:2*w] = show
        cv2.rectangle(dual, p1, p2, CROP_COLOR)
        console.show_image(dual)
        _ = cv2.waitKey(5)
        
        vals = cc[:,2]
        # if snake.size > 1:
        #     l = snake.lastn(2)
        # else:
        #     l = np.array([0,0], dtype=int) # OK ??
        def theDist(a):
            return (np.linalg.norm([a[0],a[1]] - p))
        if len(cc) > 0:
            if snake.size >= 2:
                dists = np.apply_along_axis(theDist, 1, cc)
            else:
                dists = np.zeros(len(cc), dtype='int')
            scores = np.zeros(len(cc), dtype='float')
            for i in range(0,len(cc)):
                scores[i] = 500*math.pow(scoreFormula(cc[i], snake.active, snake.size, cal.jitterDist, cal.laserIntensity, p), 2)
            if gdebug:
                # we draw the candidates with size proportional to score
                color = color_plt(VAL_COLOR)
                plt.rcParams["figure.figsize"] = [12,10]
                plt.scatter(vals, dists, marker='o', c=scores, s=scores, cmap=plt.get_cmap('Spectral'))
                cbar = plt.colorbar()
                cbar.set_label('score', rotation=270)
                plt.xlabel("Dot Intensity")
                plt.ylabel("Deviation from predicted")
                print ("Close the graph window to continue")
                plt.show()
        else:
            print ("Error: no valid candidates")

        pair[0] = background.mean()

def changeSetting(cam, prop, name, keys, key):
    k1, k2 = keys
    if key == ord(k1):
        cam.set(prop, cam.get(prop)-1.)
        print (name + "=" + str(cam.get(prop)))
    if key == ord(k2):
        cam.set(prop, cam.get(prop)+1.)
        print (name + "=" + str(cam.get(prop)))
            
def calibrateCam(cam, console):
    """Make a series of tests for calibrating laser and camera"""

    res = -1  # will get a positive value if calibration was done

    width, height = camSize(cam)
    console.reset()
    console.write ("--- LASER CALIBRATION ---")
    console.write ("Install the cam and use your laser pointer.")
    console.write ("Try to detect the laser dot on the image.")
    console.write ("You can adjust brightness (v/b) and contrast (x/c).")
    console.write ("Then switch the laser off, make sure")
    console.write ("nothing moves, and press 'q'.")
    console.write ("")
    console.write ("You may press 's' to skip calibration and use default values.")
    console.write ("Otherwise, press ESC to quit.")
    key = 0
    while True:
        img = readCam(cam)
        console.show_image(img)
        key = cv2.waitKey(10) # We could set larger than 10 to use less CPU
                              # here, but this is weird:
        # internal cam:
        # 1 ==> 23%
        # 5 ==> 23%
        # 20 ==> 24% !
        # 100 ==> 14% but latency is visible
        # 500 ==> 2%
        
        changeSetting(cam, CAM_BRIGHTNESS, "brightness", ('v','b'), key)
        changeSetting(cam, CAM_CONTRAST, "contrast", ('x','c'), key)
        changeSetting(cam, CAM_GAIN, "gain", ('f','g'), key)
        
        if key == ord('q') or key == ord('s'): break
        if key == 27: raise SystemExit
    if key == ord('s'): # return default values
        printd ("Using default values")
        return (Calibration(cam), res)
        
    # We first check light stability (due to exterior lightings or camera
    # fluctations), image stability.

    # If the room is dark one can expect more light flucutation but it's not a
    # problem because the laser will be more visible anyway. How to quantify
    # this? Learn?

    cal = Calibration(cam) # we start with default values

    console.reset()
    maxMotionThreshold = 30 # empirical: above this one can suspect global motion...
    clipBox = (0,0), (width-1, height-1) # everything
    nProbes = 20 # number of images to capture for calibration
    vals, images = getProbes(cam, nProbes, clipBox, console)
    avgVal = sum(vals)/float(len(vals))
    printd ('        Average intensity = ' + str(avgVal))
    cal.motionThreshold = int(avgVal + 0.5)
    if cal.motionThreshold <= 8: # empirical... 
        console.write ("* Light stability is good!")
    elif cal.motionThreshold >= maxMotionThreshold: #empirical...
        console.write ("# Light condition is not stable.")
        console.write ("# Maybe something has moved in the image?")
        cal.motionThreshold = maxMotionThreshold
    printd ('+ We choose motionThreshold = '+  str(cal.motionThreshold))
    gms = []
    for [bkg, img] in images:
        diff, maxVal, maxLoc = diff_max(bkg, img)
        gm = globalMotion(diff, cal.motionThreshold)
        gms.append(gm)
    cal.globalMotionThreshold = max(gms)
    printd ('        Max globalMotion = ' + str(cal.globalMotionThreshold))

    # Now we check the intensity of the laser
    # and detect size of laser beam
    console.write ('')
    console.write (" --- Switch on the laser ---")
    console.write ("Make sure the cam hasn't moved,")
    console.write ("direct the pointer to the target,")
    console.write ("and press 'q' while moving the laser pointer slowly")
    console.write ("(without stopping) inside the green box.")
    console.show()
    
    radius = min(width, height) // 4
    cx1, cy1 = width//2 - radius,     height//2 - radius
    cx2, cy2 = width//2 + radius - 1, height//2 +  radius - 1
    # we save (and crop) the last image of the previous probes
    emptyImg = images[-1][1][cy1:cy2,cx1:cx2]

    def drawFn(show):
        cv2.rectangle(show, (cx1, cy1), (cx2, cy2), CROP_COLOR, 3)
    while True:
        img = readCam(cam)
        drawFn(img)
        console.show_image(img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    clipBox = (cx1,cy1), (cx2, cy2)
    _, images = getProbes(cam, nProbes, clipBox, console, drawFn=drawFn)
    # We now compute globalMotion of the whole images, and the diff with
    # emptyImg after croping the green box (could have done this before, maybe)
    # TODO finish this
    imax = 0 # index of image with highest laser intensity
    maxi = 0 # max intensity
    i = 0
    vals, locs, gms, diams = [], [], [], [] 
    for [bkg, img] in images:

        # globalMotion
        diff, _, _ = diff_max(bkg, img)
        gm = globalMotion(diff, cal.motionThreshold)
        printd ("     globalMotion = " + str(gm))
        gms.append(gm)

        # intensity in green box
        bkg = bkg[cy1:cy2,cx1:cx2]
        img = img[cy1:cy2,cx1:cx2]
        cdiff, maxVal, maxLoc = diff_max(emptyImg, img)
        vals.append(maxVal)
        locs.append(maxLoc)
        if maxVal > maxi:
            maxi, imax = maxVal, i
        printd ("   maxVal = " +  str(maxVal))
        printd ("   maxLoc = " +  str(maxLoc))
        
        # shape in green box
        # TODO: use more that laserDiameter
        thr = cal.motionThreshold + (maxVal - cal.motionThreshold)/3
        # ou bien thr = motionThreshold ??
        _, (_,_,w,h), _ = laserShape(cdiff, maxLoc, thr, debug=True)
        diams.append(min(w,h))
        
        i +=  1

    cv2.destroyWindow("Diff")
    cv2.destroyWindow("Crop")
    cv2.destroyWindow("Mask")
    console.reset()
    
    printd ("Max intensity = " + str(maxi) + "=" + str(vals[imax]) + " at image #" + str(imax))
    
    avgVal = sum(vals)/float(len(vals))
    printd ('        Average intensity = ' + str(avgVal))
    if avgVal - cal.motionThreshold >= 30:
        console.write ("* Intensity of laser is good!")
        res = 4

    # we count the proportion of values below motionThreshold
    above = [x for x in vals if x <= cal.motionThreshold]
    if 2 * len(above) >= len(vals): # at least half of bad values (too small values)
        console.write ("# ERROR: the laser pointer was not detected. Check it and try again.")
        res = 0
    elif 10 * len(above) >= len(vals): # at least 10 of bad values (too small values)
        console.write ('# WARNING: the laser pointer is sometimes barely detectable.')
        console.write ("The detection will most probably misbehave.")
        console.write ("You should either get a brighter laser of darken the room.")
        res = 1
    elif avgVal <= cal.motionThreshold + 5:
        console.write ("# WARNING: the laser pointer is not bright enough")
        console.write ("to ensure a good detection. You should either")
        console.write ("get a brighter laser of darken the room.")
        res = 2
    else:
        res = 3

    cal.laserDiameter = sum(diams) / len(diams)
    printd ("+ Detected laser diameter = " + str(cal.laserDiameter))
    if cal.laserDiameter < 6:
        console.write ('# WARNING: the laser dot is very small.')

    console.show()
    gms.append(cal.globalMotionThreshold)
    cal.globalMotionThreshold = 5 * max(gms) #?? pourquoi 5, à vérifier
    printd ('+ We choose globalMotionThreshold = '+  str(cal.globalMotionThreshold))
    
    if gdebug:
        cv2.destroyWindow("Crop")
        cv2.destroyWindow("Mask")
        cv2.destroyWindow("Diff")

    optimize(cal, images, clipBox, console)
        
    cal.laserIntensity = avgVal
    return (cal, res)

def insideBox(z, box):
    (x,y) = z
    ((x1,y1), (x2,y2)) = box
    """Check if position is inside box. Assumes x1<=x2 and y1<=y2."""
    return (x <= x2 and x >= x1 and y <= y2 and y >= y1)

def maxValPos(gray, threshold, nmax):
    """from the gray image, return an array of [x,y, value] with the higher values, of max length nmax"""

    t0 = timeit.default_timer() 
    selecty, selectx = np.where(gray >= threshold)
    # "np.where" is quite slow... about 10x more than "val=" or "sort" below...
    # typically 0.001 sec for 200 size
    print_time ("size = " + str(len(selectx)) + " select", t0)
    t0 = timeit.default_timer()
    val = gray[gray >= threshold] # let's hope the order is the same as that
                                  # was used for selectx/y...
    print_time ("val", t0)
    
    if len(selectx) <= nmax:
        t0 = timeit.default_timer()
        res = np.column_stack((selectx,selecty,val))
        print_time ("stack", t0)
        return res
    
    else: # we need to sort... (this case should be avoided for performance)
        t0 = timeit.default_timer()
        # we create a structured array in order to sort by the value :
        a = np.zeros((len(selectx),), dtype=[('x', 'i4'), ('y', 'i4'), ('val', 'i1')])
        a['x'] = selectx
        a['y'] = selecty
        a['val'] =  val
        sorted = np.sort(a, order='val')

        # we take the last nmax elements:
        best = sorted[-nmax:]

        # and convert back to a normal array:
        res = np.zeros((nmax,3), dtype=int)
        res[:,0] = best['x']
        res[:,1] = best['y']
        res[:,2] = best['val']
        print_time ("sorting", t0)
        return res

def gaussian(x,x0,sigma):
    # do something faster ?? we don't need a precise gaussian here
  return np.exp(-np.power((x - x0)/sigma, 2.)/2.)


def scoreFormula(candidate, active, snakeSize, jitterDist, laserIntensity, predicted):
    """The best score will select the good pixel [x,y,val]"""
    # score should be a float >= 0.
    # A pixel will be selected if its score is >= 0.5
    
    # Cases where a pixel should be clearly selected (score>=1)
    # (after checking globalMotion):
    #  1. intensity (0-255) is > 40 (bright pointer) # this can be detected maybe
    #  2. pixel is very close to predicted value (regular motion)
    #  3. area is small and pixel is close to predicted value (steady pointer)

    # Cases where a pixel should NOT be selected
    #  1. snake is active and pixel is far from predicted

    #

    intensity = candidate[2] / float(laserIntensity)

    if predicted is not None:
        deviation =  np.linalg.norm(predicted - candidate[0:2])
        # TODO should use the relative deviation wrt the distance.
        wellPredicted = gaussian(deviation, 0, jitterDist)
        #return wellPredicted
        return (intensity + 5*wellPredicted)/6.
    else:
        return intensity + 0.1 # if there is no snake we need to give a bonus.

def bestPixel(candidates, active, snakeSize, jitterDist, laserIntensity, predicted):
    """return the pixel with max score"""
    # avoid sorting, since only linear complexity is necessary here
    s = 0.
    i0 = 0
    for i in range(len(candidates)):
        ss = scoreFormula(candidates[i], active, snakeSize, jitterDist,
                          laserIntensity, predicted)
        if ss > s:
            i0 = i
            s = ss
    return (candidates[i0], s)

def plotVal(show, candidate, color=VAL_COLOR, thickness=1):
    "draw a circle around the position with a radius proportional to the value"
    x,y = candidate[0],candidate[1]
    size = max(0, int(candidate[2]//4))
    cv2.circle(show, (x,y), size, color, thickness, cv2.LINE_AA)

# This is the main detection function
# snake and background are mutable, modified by this function.
def oneStepTracker(background, img, show, clipBox, snake, cal):
    global gdebug
    mask = []

    if background.empty:
        background.add(img)
        return (mask, 0)
    
    printd ("/------------------- new image --------------------\\")
    diff, maxVal, maxLoc = diff_max(background.mean(), img, cal.laserDiameter//2)
    gm = globalMotion(diff, cal.motionThreshold)
    printd ("Global Motion  = " + str(gm))
    printd ("Max Intensity  = " + str(maxVal)) # between 0 and 255

    if gdebug:
        (x,y) = maxLoc
        plotVal(show, [x,y,maxVal], color=MAXVAL_COLOR)
    
    newPoint = False
    
    # We try to detect the pointer only if there is no global motion of the
    # image:
    if maxVal > cal.motionThreshold and gm < cal.globalMotionThreshold and insideBox(maxLoc, clipBox):
        # now we detect all the points above candidateThreshold and
        # try to select the best one...
        candidateThreshold = maxVal-5 # ?? ou motionThreshold ?
        t0 = timeit.default_timer()
        candidates = maxValPos(diff, cal.motionThreshold, 10)
        print_time ("Candidates (sec)", t0)
        printd (candidates.shape)
        if gdebug:
            for c in candidates:
                printd (c)
                plotVal(show, c)
            
        if not snake.empty():
            # we compute the predicted position
            p = snake.predict()
            printd ("Predicted = " + str(p))
            if gdebug:
                cv2.circle(show, (p[0], p[1]), 10, PREDICTED_COLOR, 1)
        else:
            p = None

        # We select the candidate with the best score
        best, score = bestPixel(candidates, snake.active, snake.size, cal.jitterDist, cal.laserIntensity, p)
        printd ("SCORE = " + str(score))
        if snake.active:
            dd =  np.linalg.norm(p - best[0:2])
            printd ("Deviation from prediction = " + str(dd))
            # distance from last recorded point. Not used yet.
            d = np.linalg.norm(best[0:2] - snake.last())
            printd ("Distance = " + str(d))

        if gdebug:
            # TODO use the result of laserShape below! we could use the
            # fact that the shape often indicates the direction of the
            # pointer
            thr = cal.motionThreshold + (best[2] - cal.motionThreshold)/3
            # ou bien thr = motionThreshold ?
            mask, rect, angle = laserShape(diff, (best[0],best[1]), thr,
                                           maxRadius=int(cal.laserDiameter), debug=False)


        if score >= 0.5:
            printd ("==> Adding new point to snake.")
            newPoint = True # finally we register the best point

            if snake.size >= 1 and (np.linalg.norm(snake.last() - best[0:2]) > cal.laserDiameter):
                background.add(img)
            else:
                printd ("Not updating background because pointer did not move enough.")

            snake.grow(best[0:2])
            l = snake.length()
            a = snake.area(show)
            printd ("Length         = " + str(l))
            printd ("Area           = " + str(a))

        else:
            printd ("Nothing found.")

    snake.active = newPoint
    if (not snake.active) and snake.size > 0:
        snake.remove()

    printd ("Snake size = " + str(snake.size))
    if snake.size != 0:
        snake.draw(show)
    else:
        background.add(img)

    return (mask, maxVal)

def calibrateLoop(cam, console):
    """Repeat calibration until successful"""
    res = 0
    while res == 0:
        (cal, res) = calibrateCam(cam, console)
        if res == 0:
            console.reset()
            console.write ("## Calibration failed! Please try again ##")
            console.write ("")
            console.write ("Press any key")
            console.show()
            _ = cv2.waitKey(0)
    return (cal)

def flush(cam, n=5):
    for i in range(n):
        _ = readCam(cam)

def plot_times(profile):
    data = np.array(profile)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(data[:,0], label='one_step')
    ax.plot(data[:,1], label='total')
    plt.legend()
    plt.show()

# ------------------------------------------
# stand-alone tracker for demo and debugging
# ------------------------------------------
def webcamTracker(camera_id, debug):
    global gdebug
    gdebug = debug
    laserWindow = "Laser"
    tmpdir = tempfile.mkdtemp(prefix='laser_')
    
    # tunable variable
    snake_max_size = 10
    bkgLen = 10
    sleep_time = 10 # time of inactivity in seconds before "sleep state"

    console = Console("Console", (640,480), 240)
    cam, width, height = openCam(camera_id)
    if isinstance(camera_id, int):
        cal = calibrateLoop(cam, console)
        cal.save("%s/calibration.yml"%tmpdir)
    else:
        cal = Calibration(cam)
        print ("Loading calibration data")
        cal.load(cam, "%s/calibration.yml"%os.path.dirname(camera_id))
        
    console.write ("------------------------------------------")
    console.write (" Press any key to start the tracking session")
    console.write ("------------------------------------------")
    console.show()
    _ = cv2.waitKey(0)
    
    clipBox = (1,1), (width-2, height-2) # we remove 1 pix off every border

    # Flush webcam buffer
    t0 = timeit.default_timer()
    flushSize = 10
    flush(cam, flushSize)
    printd ("Avg FPS for reading cam=" + str(flushSize/(timeit.default_timer()-t0)))
        
    background = Background(bkgLen)
    # The background is the image that will be substracted to the current image
    # in order to detect laser motion.
    
    snake = Snake(snake_max_size)
    if debug:
        cv2.namedWindow(laserWindow, cv2.WINDOW_NORMAL)
    print ("Press 'q' to exit")
    print ("Press 'd' to toggle debugging")
    startFPS = timeit.default_timer()
    frame_count = 0
    save = False
    startSave = 0
    time_profile = []
    while True:        
        # Retrieve an image and Display it.
        t1 = timeit.default_timer()
        img = readCam(cam)
        snake.next_frame()
        show = img.copy() # this is the console image where we can draw.

        # THIS IS THE MAIN DETECTION STEP:
        t0 = timeit.default_timer()
        mask, _ = oneStepTracker(background, img, show, clipBox, snake, cal)
        tt0 = print_time("One Step", t0)
        
        if mask != []:
            cv2.imshow("Laser", mask)

        # We display the active status in the console.
        console.reset()
        if gdebug:
            console.write ("Frame #%u"%snake.current_frame)
        console.write ("q = quit; p = pause; d = toggle debug mode")
        if save:
            console.write ("Saving frame #%u in %s"%(snake.current_frame,tmpdir))
            console.write ("s = stop saving")
            cv2.imwrite ("%s/frame_%07d.jpg"%(tmpdir,snake.current_frame - startSave), img)
        else:
            console.write ("s = save all frames")
        console.write ("active = " + str(snake.active))
        console.write("button down = " + str(snake.button_down))
        if snake.click:
            console.write("==> Click! <==")
        console.show_image(show)
        tt1 = print_time("total", t1)

        # Compute FPS
        frame_count += 1
        fps = frame_count/(timeit.default_timer() - startFPS)
        printd ("FPS = %f"%fps)
        if frame_count == 1000:
            frame_count = 0
            startFPS = timeit.default_timer()

        if gdebug:
            time_profile.append(np.array([tt0,tt1,fps]))    

        # Display console and wait for key.
        idle_time = (snake.current_frame - snake.last_active_frame)/fps
        dt = (250 if idle_time > sleep_time else 50 if snake.size == 0 else 10)
        key = cv2.waitKey(dt)
        #time.sleep(dt/1000.)
        if key == ord('q'):
            break
        if key == ord('d'):
            gdebug = not gdebug
            if gdebug:
                cv2.namedWindow(laserWindow, cv2.WINDOW_NORMAL)
            else:
                cv2.destroyWindow(laserWindow)
                cv2.destroyWindow("BACKGROUND")
        if key == ord('s'):
            save = not save
            if save:
                startSave = snake.current_frame
        if key == ord('p'):
            console.write ("PAUSED. Press any key to resume")
            console.show()
            _ = cv2.waitKey(0)
        
    console.close()
    print ("---- Data saved in %s"%tmpdir)
    if gdebug:
        background.close_window()
        plot_times(time_profile)


if __name__ == "__main__":
    print ("Welcome to Laser by San Vu Ngoc, University of Rennes 1.")
    print ("This program comes with ABSOLUTELY NO WARRANTY")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="set debug mode")
    parser.add_argument("-c", "--camera", type=int, help="set camera device id")
    parser.add_argument("-i", "--input", help="load frames from specified directory instead of camera (ignored if camera is specified)")
    args = parser.parse_args()
    debug = args.debug
    input_dir = args.input
    camera_id = args.camera
    if camera_id is None:
        if input_dir is None:
            camera_id = -1
        else:
            camera_id = input_dir + "/frame_%07d.jpg"  #TODO put this in a variable

    webcamTracker (camera_id, debug)
    print ("Bye")

    
'''
0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
*3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
*4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
5. CV_CAP_PROP_FPS Frame rate.
6. CV_CAP_PROP_FOURCC 4-character code of codec.
7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
*10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
*11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
*12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
*13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
*14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
'''


# Result of the test:
# np.argwhere(a>10) is equivalent to np.transpose(np.nonzero(a>10))
# np.argwhere(a>0) is equivalent to np.transpose(np.nonzero(a>0))
# np.argwhere(a != 0) is equivalent to np.transpose(np.nonzero(a != 0))
# BUT Attention
# np.argwhere(a) is equivalent np.transpose(np.nonzero(a)) BUT is more than 2x slower than np.argwhere(a != 0)
# REMARK: if we know that a>=0 then testing a>0 is 6% FASTER than testing a!=0
def test():
    n=3000 # size of matrix
    nz=100 # number of non zero entries
    nt=200 # number of tests

    # initialize random sparse matrix
    a=np.zeros((n,n), np.int)
    for i in range(nz):
        a[randint(0,n-1), randint(0,n-1)] = randint(1,100)

    for repeat in range(5):
        
        # test argwhere
        t=timeit.default_timer() 
        for i in range(nt):
            res = np.argwhere(a > 0)
        print_time("argwhere",t,thr=0)

        # test nonzero
        t=timeit.default_timer()  
        for i in range(nt):
            res = np.transpose(np.nonzero(a > 0))
        print_time("nonzero",t,thr=0)

