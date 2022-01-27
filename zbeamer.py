# -*- coding: utf-8 -*

# ZBEAMER

# (c) 2018, San Vu Ngoc. University of Rennes 1.

# control your presentation with any laser pointer
# (or even a good focalized flash light)


# INTRO
# on affiche le message de bienvenue:
#
# 1. passage en plein écran
# 2. image du cinéma avec le message "Le petit pointeur production presents..."

# WEBCAM
# on vérifie que la webcam est branchée et est orientée vers l'écran

# on ajuste la position de la webcam en détectant le premier rectangle blanc:
# 1. affichage du fond noir, attente 0.2 sec, sauvegarde webcam
# 2. affichage rectangle blanc, attente 0.2 sec, sauvegarde webcam
# 3. lancer calibrate_rect.py avec les deux images
# 4. si pas bon, on affiche un message et on recommence en 1.
# 5. ok, à partir de maintenant toutes les images de la webcam seront
#    redressés avec le rectangle obtenu.

# on construit le noyau:
# 1. afficher un pixel blanc central sur fond noir, attendre 0.2 sec
#    et sauvegarde webcam redressée
# 2. soustraire le fond noir déjà enregistré ? ce qui devrait donner
#    une approx de la solution fondamentale
# 3. on calcule la FFT  (fn de transfert)
#    ? Faut-il essayer à d'autres endroits?

# on peut commencer la présentation et détecter le laser:
# 1. afficher l'image en cours
# 2. régulièrement, sauvegarder la webcam et redresser.
# 3. soustraire le noir (NON, pas bon), FFT, diviser par la fn de transfert,
#    FFT inverse (ou bien dans l'autre sens: simuler la projection de l'image:
#    convoler avec le noyau: (opencv utilise la FFT pour convoler dès que le
#    noyau est gros)



# REM: en mode "image statique" on n'a pas besoin de déconvolution, on peut
# supposer que le laser est éteint à chaque changement d'image et donc
# sauvegarder l'image dès qu'elle arrive pour faire les différences

# REM: pour bouger la souris:
# xdotool mousemove <x> <y>
# ou
# https://pyautogui.readthedocs.io/en/latest/introduction.html#purpose
# sudo apt install python-xlib 
# sudo pip install pyautogui

# other related projects:
# http://www.cs.technion.ac.il/~zachik/presentermouse/index.htm
# http://web.mit.edu/6.111/www/s2006/PROJECT/3/Project3.pdf
# http://www.cs.columbia.edu/~hgs/research/projects/laserpointer-mouse/
# http://www.cs.binghamton.edu/~reckert/chi2000fin4.PDF


import sys
import timeit

import numpy as np
import cv2 as cv

import laser9 as laser
#import os.path

print "OpenCV version :  {0}".format(cv.__version__)

debug = True
fake = False

def debugWait (s=""):
    "Wait for a key in debug mode, otherwise wait 5ms."
    if debug:
        print(s)
        print("Press a key")
        _ = cv.waitKey(0)
    else:
        _ = cv.waitKey(5)

def gaussianBeam(radius, color):
    "return a square image of width 2*radius+1"
    w = 2*radius+1
    img = np.zeros((w,w,3), np.uint8)
    cv.circle(img, (radius, radius), radius/2, color, -1) # filled circle
    blr = cv.GaussianBlur(img, (w,w),0) # probably not necessay since we blur for detection anyway
    return (blr)

def gaussianPointer(image, (x,y), radius, color):
    "draw a light spot on the image"
    beam = gaussianBeam(radius, color)
    mask = gaussianBeam(radius, (255,255,255)) # of course we could optimize this
    mask = 1 - mask.astype(float)/255
    w = 2 * radius + 1
    imgh, imgw, _ =  image.shape

    x1 = x - radius
    if x1 < 0:
        dx1 = -x1
    else:
        dx1 = 0

    y1 = y - radius
    if y1 < 0:
        dy1 = -y1
    else:
        dy1 = 0    

    x2 = x + radius
    if x2 + 1 > imgw:
        dx2 = x2 + 1 - imgw
    else:
        dx2 = 0

    y2 = y + radius
    if y2 + 1 > imgh:
        dy2 = y2 + 1 - imgh
    else:
        dy2 = 0    

    print(y1 + dy1, y2 + dy2 + 1, x1 + dx1, x2 - dx2 + 1)
    crop = image[y1 + dy1:y2 + dy2 + 1, x1 + dx1:x2 - dx2 + 1]
    mask = mask[dy1:w-dy2, dx1:w-dx2]
    beam = beam[dy1:w-dy2, dx1:w-dx2]
    if mask.shape != crop.shape:
        print (str(mask.shape) + " should equal " + str(crop.shape))
    blend = np.multiply(crop, mask).astype(np.uint8)
    crop = cv.add(blend, beam)
    image[y1 + dy1:y2 + dy2 + 1, x1 + dx1:x2 - dx2 + 1] = crop


        
class Projector:
    def __init__(self, name, console, aspect=(4,3)):
        self.name = name
        self.aspect = aspect # aspect ratio of the projector
        # initialize window
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        a, b = aspect
        width, height = 640, 640 * b / a
        cv.resizeWindow(name, width, height)
        # initialize image (black)
        self.image = np.zeros((height, width, 3), np.uint8)
        # initialize console
        self.console = laser.Console(console, (width, height), height/2)

    def size(self):
         # Size of the image on the external screen. It is the size used to
         # compute the perspective transformation. But it's not necessarily the
         # resolution of the external monitor (the image is scaled to
         # fullscreen).
        h,w,_ = self.image.shape
        return (w,h)
    
    def show(self):
        cv.imshow(self.name, self.image)
        # WARNING from opencv doc: Note This function (imshow) should be
        # followed by waitKey function which displays the image for specified
        # milliseconds. Otherwise, it won’t display the image.

    def set_image(self, image):
        self.image = image

    def show_image(self, image):
        self.set_image(image)
        self.show()
        
    def show_console(self, image):
        self.console.set_image(image)
        self.console.show()

    def close(self):
        cv.destroyWindow(self.name)
        self.console.close()
        
        
# INTRO
#######

def equal_rat ((a,b),(u,v)):
    return (a*v == b*u)

def positionWindows(fontScale=1.5):
    "Create projector and ask user for moving windows"
    
    proj = Projector("Projector", "Console")
    proj.console.fontScale = fontScale

    proj.console.write("------------------")
    proj.console.write("Welcome to zbeamer")
    proj.console.write("------------------")
    proj.console.write("This is your console, keep it near you.")
    proj.console.write("Move the other window to the display")
    proj.console.write("that will be sent to the projector.")
    proj.console.write("Press 'f' to set that window to fullscreen mode.")
    proj.console.write("")
    proj.console.write("When you are done, press 'q'")
    proj.console.show()
    

    # load theater background
    if equal_rat (proj.aspect, (4,3)):
        file = "theater-4:3.jpeg"
        screen = {
            'size' : (1024, 768),
            # 'size' is the size of the theater image that is displayed
            # fullscreen on the external monitor. It is not necessarily the
            # same resolution as the monitor, because it will be scaled.
            'w' : 309,
            'h' : 212,
            'x' : 357,
            'y' : 231
            }
    else:
        print("Only (4/3) aspect ratio is currently supported")
        sys.exit(1)

    bg = cv.imread(file)
    bg = cv.resize(bg, screen['size'])

    proj.set_image(bg)
    zb = cv.imread("zbeamer_logo.jpg")
    theaterScreen(screen, proj.image, zb)
    proj.show_console(zb)
    proj.show()

    # put it fullscreen

    fullscreen = False
    while True:
        k = cv.waitKey(0)
        if (k == ord('q')):
            break
        if (k == ord ('f')):
            if fullscreen:
                cv.setWindowProperty(proj.name,cv.WND_PROP_FULLSCREEN,
                                     cv.WINDOW_NORMAL)
            else:
                cv.setWindowProperty(proj.name,cv.WND_PROP_FULLSCREEN, 1)
            fullscreen = not fullscreen

    return(proj, screen)

    # TODO show trailer video

# WEBCAM
########

def theaterScreen(screen, bg, image):
     # put image inside theater screen
     resized = cv.resize(image, (screen['w'], screen['h']))
     bg[screen['y']:screen['y']+screen['h'], screen['x']:screen['x']+screen['w']] = resized

def setupWebcam(proj, screen):
    # config variable:
    camera = 0 # use 0 for internal webcam (if no external webcam was plugged-in
                # when the computer started up), -1 for automatic detection
    cam, w, h = laser.openCam(camera)

    proj.console.reset()
    proj.console.write("Direct the webcam towards the conference screen")
    proj.console.write("so that you can see the whole theater image")
    proj.console.write("in the small theater screen at the middle")
    proj.console.write("('ad infinitum').")
    proj.console.write("")
    proj.console.write("Press 'q' when you are done.")
    proj.console.show()
    _ = cv.waitKey(1)
    
    # show webcam inside theater screen
    while True:
        img = laser.readCam(cam)
        if(cv.waitKey(16)==ord('q')):
            break

        proj.console.set_image(img)
        #resized = cv.resize(img, (screen['w'], screen['h']))
        bg = proj.image
        theaterScreen(screen, bg, img)
        #bg[screen['y']:screen['y']+screen['h'], screen['x']:screen['x']+screen['w']] = resized
        proj.show()

    return(cam)

def project(cam, proj, img, frames=5):
    """
    Send img to projector, and return before/after webcam images.
    Optionnally wait some delay before recording the new image to deal
    with cam latency. frames=5 should be enough for this. Waiting longer
    may help the camera adjust to new image brightness (don't do this if
    you want to use the difference between before and after.)
    """
    bg = laser.readCam(cam)
    
    proj.show_image(img)
    _ = cv.waitKey(2) # 2
    m = 0
    imax = 0
    recmax = bg
    for i in range(frames):
        rec = laser.readCam(cam)
        proj.show_console(rec)
        _ = cv.waitKey(4)
        _, maxVal, _ = laser.diff_max(bg, rec)
        if maxVal > m:
            imax = i
            m = maxVal
            recmax = rec
        print ("maxVal[" + str(i) + "] = " + str(maxVal))

    print ("maximum difference (" + str(m) + ") was obtained at frame#" + str(imax))
    return (bg, recmax)

def detectGaussian(cam, proj, img, (x,y), radius):
    """
    Send a Gaussian beam to the projector and detect it back from the cam.
    This does not modify img.
    """
    # we use black or white depending on background
    h,w,_ = img.shape
    crop = img[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    mean = cv.mean(gray)[0]
    print ("mean="+str(mean))
    if mean < 127:
        c = 255
    else:
        c = 0
    image = img.copy()
    # first we project the black image
    bkg, rec = project(cam, proj, image)
    # and then the pointer image
    gaussianPointer(image, (x,y), radius, (c,c,c))
    bkg, rec = project(cam, proj, image)
    if c == 0:
        bkg, rec = rec, bkg # we swap so that the difference should always be
                            # positive at the light beam.
    diff, maxVal, maxLoc = laser.diff_max(bkg, rec)
    print ("Light detected at " + str(maxLoc) + " with intensity " + str(maxVal))
    # TODO use laser.calibration
    gaussianPointer(rec, maxLoc, radius, (34,23,240))
    proj.show_console(rec)
    #debugWait("Detected light")
    return (maxLoc)

def detectScreen(cam, proj, scale=0.75):
    w,h = proj.size()
    img = np.zeros((h,w,3), np.uint8)
    xmargin = int(w * ((1. - scale)/2.))
    ymargin = int(h * ((1. - scale)/2.))
    radius = 30 # config ?
    image = proj.image
    p1 = detectGaussian(cam, proj, img, (xmargin, ymargin), radius)
    p2 = detectGaussian(cam, proj, img, (w - xmargin, ymargin), radius)
    p3 = detectGaussian(cam, proj, img, (w - xmargin, h - ymargin), radius)
    p4 = detectGaussian(cam, proj, img, (xmargin, h - ymargin), radius)
    proj.show_image(image)
    return (np.array([p1,p2,p3,p4]))
            

def screenDetection(cam, proj):

    # config variable
    scale = 0.75 # size of the detection screen in percentage

    ok = False

    while not ok:
        rect = []
        while (rect == []):
            proj.console.reset()
            proj.console.write("Trying to detect the screen corners.")
            proj.console.write("Please don't move the camera or the mouse,")
            proj.console.write("and don't let anything interfere.")
            proj.console.write("")
            proj.console.show()
            rect = detectScreen(cam, proj, scale)
            print(rect)

        #debugWait("This is the detected region.")
        #rect = rct.order_rect(rect)
        if fake:
            img = cv.imread("room-white.jpg")
        else:
            img = laser.readCam(cam)
        cnt = rect.reshape(4,1,2).astype(int)
        cv.drawContours(img, [cnt], 0, (255, 255, 12), 5)
        for p in rect:
            gaussianPointer(img, (p[0],p[1]), 10, (24,50,255))
        proj.console.write("Was the screen area correctly detected (y/n)?")
        proj.show_console(img)
        k = 0
        while k != ord('y') and k != ord('n') and k != 27:
            k = cv.waitKey(0)
        ok = (k == ord('y'))

    print("Great!")
    proj.console.reset()
    proj.console.show()
    return(rect, scale)

# show the screen zone in the console:

def getProjPerspective(proj, rect, scale):
    """Compute the perspective transform matrix.

    This is a 3x3 float matrix, giving the map from the image grabbed by
    the webcam, to the proj.image. So be careful of scaling issues when
    you change the size of any of these.
"""
    
    width, height = proj.size()
    xmargin = int(width * ((1. - scale)/2.))
    ymargin = int(height * ((1. - scale)/2.))
    dst = np.array([
        [xmargin, ymargin],
        [width - xmargin - 1, ymargin],
        [width - xmargin - 1, height - ymargin - 1],
        [xmargin, height - ymargin - 1]], dtype = "float32")
    M = cv.getPerspectiveTransform(rect.astype('float32'), dst)
    return (M)

def redressCam(cam, M, console):
    """Show the corrected webcam image onto the console

    The transform M is used to redresse the webcam image.
    """
    
    console.write("You should see now the redressed screen image.")
    console.write("")
    console.write("Press 'q' to continue.")
    console.show()
    while True:
        if fake:
            img = cv.imread("room-white.jpg")
        else:
            img = laser.readCam(cam)
        flat = cv.warpPerspective(img, M, console.size)
        console.set_image(flat)
        console.show()
        if(cv.waitKey(16)==ord('q')):
            console.reset()
            break

# plot a blurred gaussian at the transformed position
def plot(proj, point, M):
    s = point.reshape(1,1,2).astype('float32')
    p = cv.perspectiveTransform(s, M)[0][0].astype('int')
    z = (p[0],p[1])
    if laser.insideBox(z, ((0,0), proj.size())):
        gaussianPointer(proj.image, z, 20, (34,250,250))

        
def followLaser(cam, proj, cal, M):
    w,h = laser.camSize(cam)
    clipBox = (1,1), (w-2, h-2) # we remove 1 pix off every border
    snakeMaxSize = 10
    bkgLen = 10
    snake = laser.Snake(snakeMaxSize)
    bkg = laser.Background(bkgLen)
    print("Follow laser... press a key")
    _ = cv.waitKey(0)
    print ("Starting detection loop")
    slide = proj.image.copy()
    dt = 0
    proj.console.reset()
    proj.console.write ("Press 'q' to quit.") 
    while True:
        img = laser.readCam(cam) # fast, 0.001
        key = cv.waitKey(max(4, 17-dt))
        # well, dt is always larger than 17 anyway...
        if key == ord('q'):
            break

        t0 = timeit.default_timer()
        proj.set_image(slide.copy())
        show = img.copy() # fast, 0.001 to 0.003

        mask, _ = laser.oneStepTracker(bkg, img, show, clipBox, snake, cal) # slow0.01
        
        if not snake.empty():
            # PERFORMANCE NOTICE: writing and updating the console is quite
            # heavy.  Disable this (or *reduce console size*) in case of too
            # high CPU usage.
            proj.console.reset() # very fast, 0.0002
            proj.console.write ("Press 'q' to quit.") 
            proj.console.write ("active = " + str(snake.active)) # fast 0.006
            proj.show_console(show) # slow, 0.01
            vsb = snake.visible().astype('float32')
            vsb = vsb.reshape(len(vsb),1,2)
            pts = cv.perspectiveTransform(vsb, M).astype('int')
            cv.polylines(proj.image, [pts], False, (234,123,123), 1, cv.CV_AA)
            # fast or very fast, 0.001 to 0.0005 depending on image size
            if snake.active:
                plot(proj, snake.last(), M) # 0.001 or less
            proj.show() # slow, 0.01 to 0.04
            # performance of course heavily depends on image size.
            dt = int(1000 * laser.print_time("Total step", t0)) # about 40ms...

        
def main():
    print("Hello")
    proj, screen = positionWindows()
    cam = setupWebcam(proj, screen)
    rect, scale = screenDetection(cam, proj)
    M = getProjPerspective(proj, rect, scale)
    redressCam(cam, M, proj.console)

    laser.gdebug = False
    proj.image[:,:,:] = 255 # plutôt: projeter un carré correspondant au vert !
    proj.show()
    cal = laser.calibrateLoop(cam, proj.console)
    followLaser(cam, proj, cal, M)
    
    cv.setWindowProperty(proj.console.name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
    proj.close()
    cam.release()
    print "Bye"


if __name__ == "__main__":
    # execute only if run as a script
    main()
    
