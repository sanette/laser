# -*- coding: utf-8 -*

# Laser pointer detector, V1.
# ---------------------------

# Use a webcam to detect the position of the light beam emitted by a laser
# pointer, or by any strongly focused lamp.

# Copyright (C) 2018 San Vu Ngoc
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

# This version uses python2 and opencv 2.4

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import yaml
import timeit
import argparse

if cv2.__version__.startswith("3."):
    cv2.CV_AA = cv2.LINE_AA
    
# Global debugging variable
gdebug = True

# BGR colors
maxValColor = (0,255,230) # yellow
predictedColor = (155,45,0) # dark blue
valColor = (45,23,240) # bright red
cropColor = (12,234,54) # green
snakeColor = (5,200,160) # yellow-green

# convert openCV color to mathplotlib color
def colorPlt((b,g,r)):
    return (np.array((r/255.0, g/255.0, b/255.0)))
    
# Some debugging utilities
def printd(s):
    global gdebug
    if gdebug:
        print(s)

def printTime(s, t0, thr=0.0001):
    global gdebug
    t = timeit.default_timer() - t0
    if gdebug and t >= thr:
        print ("TIME " + s + " = " + str(t))
    return (t)


# At the heart of the algorithm we use background signed substraction for a
# fast "motion" detector.
#
# Warning, the resulting diff is not a valid openCV image, it can contain
# negative integers
def diffMax(img1, img2, radius=2):
    """signed difference img2 - img1 and max value and position"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # We blur the images by radius at least 3. For 640x480 webcam, 3 seems to
    # substancially improve laser detection.
    radius = 2*(max(2, radius)/2)+1 # should be an odd number
    gray1 = cv2.GaussianBlur(gray1, (radius, radius), 0)
    gray2 = cv2.GaussianBlur(gray2, (radius, radius), 0)
    h,w = gray1.shape
    diff = gray2.astype(int) - gray1.astype(int)
    imax = np.argmax(diff)
    x, y = (imax % w, imax / w)
    maxVal = diff[y,x]
    maxLoc = (x,y)
    return (diff, maxVal, maxLoc)
        
def getAngle(p1,p2):
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
        det = np.linalg.det([p1,p2])
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
        self.fontScale = 1.5
        self.thickness = 2
        self.color = (5,15,67)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w, h + textheight)
        self.image = np.zeros((h, w, 3), np.uint8)
        self.cons =  np.ones((textheight, w, 3), np.uint8) * 255
        self.total = np.zeros((h+textheight, w, 3), np.uint8)
        
    def show(self):
        ih,iw,_ = self.image.shape
        th,tw,_ = self.cons.shape
        if iw != tw:
            print ("Error text width " + str(tw)+" and image width " + str(iw) + "should be equal")
        self.total[0:ih,:] = self.image
        self.total[ih:ih+th,:] = self.cons
        cv2.imshow(self.name, self.total)

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
        
    def write(self, text):
        """Write one line of text to the console and advance line"""
        (tw,th), baseVal = cv2.getTextSize(text, self.font, self.fontScale, self.thickness)
        w,h = self.size
        if tw + self.leftmargin > w:
            scale = self.fontScale*w/float(tw + self.leftmargin)
        else:
            scale = self.fontScale
        printd ("[CONSOLE]: " + text)
        cv2.putText(self.cons, text, (self.leftmargin, self.hline + th),
                    self.font, scale, self.color, self.thickness,
                    cv2.CV_AA, False)
        self.hline = self.hline + th + baseVal

    def reset(self):
        """Clear text area (but leaves the image area)"""
        self.cons[:,:,:] = 255
        self.hline = self.topmargin

    def close(self):
        cv2.destroyWindow(self.name)

class Snake:
    """Class for snake operations

    The 'snake' is the list of last contiguously detected pixels,
    like in the classic 'snake' game.
    """

    def __init__(self, snakeMaxSize):
        self.maxSize = snakeMaxSize
        self.points = np.zeros((snakeMaxSize,1,2), dtype=int)
        # we start from the end of the array... don't ask why.
        self.size = 0
        self.active = False # is the light on?
        
    def visible(self):
        """return the visible part"""
        start = len(self.points) - self.size
        return (self.points[start:])

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

    def predict(self):
        """predict the next point by 2nd order curvature approx"""
        # about 1e-04 sec, why so "slow"?
        snake = self.points
        if self.size == 0:
            # cannot predict
            return (np.array([0,0], dtype=int))
        elif self.size == 1:
            return (snake[-1][0])
        elif self.size == 2:
            # first order
            l = snake[-1] - snake[-2]
            return ((snake[-1] + l)[0])
        else: #second order
            l1 = snake[-2] - snake[-3]
            l2 = snake[-1] - snake[-2]
            a = getAngle(l2[0], l1[0]) * 180 / math.pi
            rot = cv2.getRotationMatrix2D((0,0), a, 1) # shape (2,3)
            r = np.matmul(rot,np.array([l2[0][0],l2[0][1],0]))
            return (snake[-1][0] + r.astype(int))

    def grow(self, point):
        """add a point to the snake"""
        # we add to its end, shifting the others to the left:
        snake = self.points
        l = len(snake)
        snake[0:l-1] = snake[1:l]
        snake[l-1] = point
        if self.size == l:
            s = self.size
        else:
            s = self.size + 1
        self.size = s

    def remove(self):
        """remove the first point = tail of the snake"""
        if self.size == 0:
            print ("ERROR: cannot remove point of empty snake")
        else:
            self.size = self.size - 1

    def draw(self, image):
        """draw the snake on the image"""
        if self.size != 0:
            cv2.polylines(image, [self.visible()], False, snakeColor, 2,
                          cv2.CV_AA)
            if self.active:
                p = self.last()
                cv2.circle(image, (p[0], p[1]), 5, snakeColor, -1, cv2.CV_AA)

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
    return (len(moved) / float (w*h))


# find connected component of initial point maxLoc of pixels with gray value >=
# threshold.
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
    if cv2.__version__.startswith("3."):
        retval, _, mask, rect = cv2.floodFill(crop, mask, seed, 125, lowDiff, 255, cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | 4 | ( 255 << 8 ) )
    else:
        retval, rect = cv2.floodFill(crop, mask, seed, 125, lowDiff, 255, cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | 4 | ( 255 << 8 ) )
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
        
        
    return (mask, rect)

# Return a list of (nProbes-1) pairs of images [background, image], and the
# list of maxVals
def getProbes(cam, nProbes, clipBox, console, drawFn=None):
    # cf webcamTracker for explanation of the variables
    vals = []
    images = []
    background = []
    cal = Calibration(cam)
    snake = Snake(nProbes-1)
    printd ("Please wait...")
    for i in range(nProbes):
        img = readCam(cam)
        if i != 0: # ie. if background != []
            images.append([background,img])
        show = img.copy()
        if drawFn != None:
            drawFn(show)
        _ = cv2.waitKey(10)
        background, mask, maxVal = oneStepTracker(background, img, show, clipBox, snake, cal)
        if i != 0:
            vals.append(maxVal)
        console.show_image(show)
    return (vals, images)

def camSize(cam):
    return (int(cam.get(3)), int(cam.get(4)))

def openCam(cameraId):
    cam = cv2.VideoCapture(cameraId)
    if not cam.isOpened():
        print ("Could not open Camera #" + str(cameraId))
        print ("Trying another one...")
        cam = cv2.VideoCapture(-1)
        if not cam.isOpened():
            print ("All webcam failed. Quitting.")
            exit()
    width, height = camSize(cam) 
    print ("Cam opened with size = " + str(width) + ", " + str(height))
    return (cam, width, height)

def readCam(cam):
    "Read image from webcam."
    retval, img = cam.read()
    if not retval:
        print "Cannot capture frame device"
        img = cv2.imread("webcam_error.jpeg")
    return (img)

# Class to store calibration values
class Calibration:
    def __init__(self, cam):
        w, h = camSize(cam)
        self.width, self.height = w, h
        self.diag = math.sqrt(w*w + h*h)
        # default values:
        self.motionThreshold = 7
        # = difference of intensity above which we consider moved pixels.
        self.globalMotionThreshold = 0.001
        # = percentage of moved pixels above which we decide that a global
        # motion has occured (and thus the image should not be analyzed).
        self.laserIntensity = 40
        # Reaching this intensity qualifies for being the laser beam.
        self.laserDiameter = 5 # size of laser dot in pixels.
        self.jitterArea = 0.003 # jitter area due to shaky hand.
        self.jitterDist = math.sqrt(self.jitterArea * h * w) # here 30

    def save(self, filename): # uses yaml module
        """Save calibration data to file"""
        with open(filename, 'w') as outfile:
            yaml.dump(self, outfile)

    def load(self, filename):
        """Update calibration data by loading file"""
        with open(filename, 'r') as stream:
            try:
                x = yaml.load(stream)
                for tag in ['width', 'height', 'diag', 'motionThreshold', 'globalMotionThreshold', 'laserIntensity', 'laserDiameter', 'jitterArea', 'jitterDist' ]:
                    setattr(self, tag, getattr(x, tag))
            except yaml.YAMLError as exc:
                print(exc)
    
def calibrateCam(cam, console):
    """Make a series of tests for calibrating laser and camera"""

    res = -1  # will get a positive value if calibration was done

    width, height = camSize(cam)
    cal = Calibration(cam) # we start with default values
    cal.save("foo")
    console.reset()
    console.write ("--- LASER CALIBRATION ---")
    console.write ("Have your laser pointer ready.")
    console.write ("Install the cam, make sure nothing moves")
    console.write ("in the image, and press 'q'.")
    console.write ("")
    console.write ("You may press 's' to skip calibration altogether")
    console.write ("and use default values.")
    console.write ("")
    console.write ("Otherwise, press ESC to quit.")
    key = 0
    while True:
        img = readCam(cam)
        console.show_image(img)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('s'): break
        if key == 27: raise SystemExit
    if key == ord('s'): # return default values
        printd ("Using default values")
        return (cal, res)
        
    # We first check light stability (due to exterior lightings or camera
    # fluctations), image stability.

    console.reset()
    maxMotionThreshold = 30 # empirical: above this one can suspect global motion...
    clipBox = (0,0), (width-1, height-1) # avoid boundary
    nProbes = 20 # number of images to capture for calibration
    vals, images = getProbes(cam, nProbes, clipBox, console)
    avgVal = sum(vals)/float(len(vals))
    printd ('        Average intensity = ' + str(avgVal))
    cal.motionThreshold = int(avgVal + 0.5)
    if cal.motionThreshold <= 8: # empirical... 
        console.write ("* Light stability is good!")
    elif cal.motionThreshold >= maxMotionThreshold: #empirical...
        console.write ("# Light condition is not stable. Maybe something moved in the image?")
        cal.motionThreshold = maxMotionThreshold
    printd ('+ We choose motionThreshold = '+  str(cal.motionThreshold))
    gms = []
    for [bkg, img] in images:
        diff, maxVal, maxLoc = diffMax(bkg, img)
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
    
    radius = min(width, height) / 4
    cx1, cy1 = width/2 - radius,     height/2 - radius
    cx2, cy2 = width/2 + radius - 1, height/2 +  radius - 1
    # we save (and crop) the last image of the previous probes
    emptyImg = images[-1][1][cy1:cy2,cx1:cx2]

    def drawFn(show):
        cv2.rectangle(show, (cx1, cy1), (cx2, cy2), cropColor, 3)
    while True:
        img = readCam(cam)
        drawFn(img)
        console.show_image(img)
        key = cv2.waitKey(16)
        if key == ord('q'):
            break
    clipBox = (cx1,cy1), (cx2, cy2)
    _, images = getProbes(cam, nProbes, clipBox, console, drawFn=drawFn)
    # We now compute globalMotion of the whole images, and the diff with
    # emptyImg after croping the green box (could have done this before, maybe)
    imax = 0 # index of image with highest laser intensity
    maxi = 0 # max intensity
    i = 0
    vals, locs, gms, diams = [], [], [], [] 
    for [bkg, img] in images:

        # globalMotion
        diff, _, _ = diffMax(bkg, img)
        gm = globalMotion(diff, cal.motionThreshold)
        printd ("     globalMotion = " + str(gm))
        gms.append(gm)

        # intensity in green box
        bkg = bkg[cy1:cy2,cx1:cx2]
        img = img[cy1:cy2,cx1:cx2]
        cdiff, maxVal, maxLoc = diffMax(emptyImg, img)
        vals.append(maxVal)
        locs.append(maxLoc)
        if maxVal > maxi:
            maxi, imax = maxVal, i
        printd ("   maxVal = " +  str(maxVal))
        printd ("   maxLoc = " +  str(maxLoc))
        
        # measure laser size
        thr = cal.motionThreshold + (maxVal - cal.motionThreshold)/3
        # ou bien thr = motionThreshold ??
        _, (_,_,w,h) = laserShape(cdiff, maxLoc, thr, debug=False)
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

    cal.laserIntensity = avgVal
    return (cal, res)

def insideBox((x,y), ((x1,y1), (x2,y2))):
    """Check if position is inside box. Assumes x1<=x2 and y1<=y2."""
    return (x <= x2 and x >= x1 and y <= y2 and y >= y1)

def maxValPos(gray, threshold, nmax):
    """from the gray image, return an array of [x,y, value] with the higher values, of max length nmax"""

    t0 = timeit.default_timer() 
    selecty, selectx = np.where(gray >= threshold)
    printTime ("size = " + str(len(selectx)) + " select", t0)
    t0 = timeit.default_timer()
    val = gray[gray >= threshold]
    printTime ("val", t0)
    
    if len(selectx) <= nmax:
        t0 = timeit.default_timer()
        res = np.column_stack((selectx,selecty,val))
        printTime ("stack", t0)
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
        printTime ("sorting", t0)
        return res

def gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/sigma, 2.)/2.)


def scoreFormula(candidate, active, snakeSize, jitterDist, laserIntensity, deviation):
    """The best score will select the good pixel [x,y,val]"""
    # score should be a float >= 0.
    # A pixel will be selected if its score is >= 0.5

    intensity = candidate[2] / float(laserIntensity)
    wellPredicted = gaussian(deviation, 0, jitterDist)

    if snakeSize > 0:
        return (intensity + wellPredicted)/2.
    else:
        return intensity + 0.1 # if there is no snake we need to give a bonus.

def bestPixel(candidates, active, snakeSize, jitterDist, laserIntensity, deviation=0):
    """return the pixel with max score"""
    # avoid sorting, since only linear complexity is necessary here
    i = 0
    s = 0.
    i0 = 0
    while i < len(candidates):
        ss = scoreFormula(candidates[i], active, snakeSize, jitterDist, laserIntensity, deviation)
        if ss > s:
            i0 = i
            s = ss
        i = i + 1
    return (candidates[i0], s)

def plotVal(show, candidate, color=valColor, thickness=1):
    "draw a circle around the position with a radius proportional to the value"
    x,y = candidate[0],candidate[1]
    size = max(0, int(candidate[2]/4))
    cv2.circle(show, (x,y), size, color, thickness, cv2.CV_AA)

# This is the main detection function
def oneStepTracker(background, img, show, clipBox, snake, cal):
    global gdebug
    mask = []
    
    if background == []:
        return (img, mask, 0)
    
    printd ("/------------------- new image --------------------\\")
    diff, maxVal, maxLoc = diffMax(background, img, cal.laserDiameter/2)
    gm = globalMotion(diff, cal.motionThreshold)
    printd ("Global Motion  = " + str(gm))
    printd ("Max Intensity  = " + str(maxVal)) # between 0 and 255

    if gdebug:
        (x,y) = maxLoc
        plotVal(show, [x,y,maxVal], color=maxValColor)
    
    newPoint = False
    
    # We try to detect the pointer only if there is no global motion of the
    # image:
    if maxVal > cal.motionThreshold and gm < cal.globalMotionThreshold and insideBox(maxLoc, clipBox):
        # now we detect all the points above candidateThreshold and
        # try to select the best one...
        candidateThreshold = maxVal-5 # ?? ou motionThreshold ?
        t0 = timeit.default_timer()
        candidates = maxValPos(diff, cal.motionThreshold, 10)
        printTime ("Candidates (sec)", t0)
        printd (candidates.shape)
        if gdebug:
            for c in candidates:
                printd (c)
                plotVal(show, c)

        if snake.active:
            # distance from last recorded point
            d = np.linalg.norm(maxLoc - snake.last())
            printd ("Distance = " + str(d))

            # is the new point far from predicted?
            p = snake.predict()
            printd ("Predicted = " + str(p))
            if gdebug:
                cv2.circle(show, (p[0], p[1]), 10, predictedColor, 1) 
            dd =  np.linalg.norm(p - maxLoc)
            printd ("Deviation from prediction = " + str(dd))
        else:
            dd = cal.diag # ??
            d = cal.jitterDist

        # We select the candidate with the best score
        best, score = bestPixel(candidates, snake.active, snake.size, cal.jitterDist, cal.laserIntensity, deviation=dd)
        printd ("SCORE = " + str(score))

        if gdebug:
            thr = cal.motionThreshold + (best[2] - cal.motionThreshold)/3
            mask, _ = laserShape(diff, (best[0],best[1]), thr, maxRadius=cal.laserDiameter, debug=False)

        if score >= 0.5:
            printd ("==> Adding new point to snake.")
            newPoint = True # finally we register the best point

            if snake.size >= 1 and (np.linalg.norm(snake.last() - best[0:2]) > cal.laserDiameter):
                background = img
            else:
                printd ("Not updating background because pointer did not move enough.")

            snake.grow(best[0:2])

        else:
            printd ("Nothing found.")

    snake.active = newPoint
    if (not snake.active) and snake.size > 0:
        snake.remove()

    printd ("Snake size = " + str(snake.size))
    if snake.size != 0:
        snake.draw(show)
    else:
        background = img

    return (background, mask, maxVal)

def calibrateLoop(cam, console):
    """Repeat calibration until successful"""
    res = 0
    while res == 0:
        (cal, res) = calibrateCam(cam, console)
        if res == 0:
            print ("######### Calibration unsuccessful. Please try again. #########")
    return (cal)


# ------------------------------------------
# stand-alone tracker for demo and debugging
# ------------------------------------------
def webcamTracker(cameraId, debug):
    global gdebug
    gdebug = debug
    laserWindow = "Laser"
    
    # tunable variable
    snakeMaxSize = 10

    console = Console("Console", (640,480), 240)
    cam, width, height = openCam(cameraId)
    cal = calibrateLoop(cam, console)
    console.write ("------------------------------------------")
    console.write (" Press any key to start the tracking session")
    console.write ("------------------------------------------")
    console.show()
    _ = cv2.waitKey()
    
    clipBox = (1,1), (width-2, height-2) # we remove 1 pix off every border

    # Flush webcam buffer
    for i in range(10):
        _ = readCam(cam)
    
    # initialization with dummy values:
    background = []
    # The background is the image that will be substracted to the current image
    # in order to detect laser motion.
    
    snake = Snake(snakeMaxSize)
    dt = 0
    if debug:
        cv2.namedWindow(laserWindow, cv2.WINDOW_NORMAL)
    print "Press 'q' to exit"
    print "Press 'd' to toggle debugging"
    while True:        
        # Retrieve an image and Display it.
        t1 = timeit.default_timer()
        img = readCam(cam)                
        show = img.copy() # this is the console image where we can draw.

        # THIS IS THE MAIN DETECTION STEP:
        t0 = timeit.default_timer()
        background, mask, _ = oneStepTracker(background, img, show, clipBox, snake, cal)
        printTime("One Step", t0)
        
        if mask != []:
            cv2.imshow("Laser", mask)

        # We display the active status in the console.
        console.reset()
        console.write ("q = quit;  d = toggle debug mode")
        console.write ("active = " + str(snake.active))
        console.show_image(show)
        dt = int(1000 * printTime("total", t1))
        print (dt)
        
        key = cv2.waitKey(max(17-dt, 2))
        if key == ord('q'):
            break
        if key == ord('d'):
            gdebug = not gdebug
            if gdebug:
                cv2.namedWindow(laserWindow, cv2.WINDOW_NORMAL)
            else:
                cv2.destroyWindow(laserWindow)

        
    console.close()


if __name__ == "__main__":
    print "Welcome to Laser by San Vu Ngoc, University of Rennes 1."
    print "This program comes with ABSOLUTELY NO WARRANTY"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="set debug mode")
    parser.add_argument("-c", "--camera", type=int, help="set camera device id")
    args = parser.parse_args()
    debug = args.debug
    cameraId = args.camera
    if cameraId == None:
        cameraId = -1
    webcamTracker (cameraId, debug)
