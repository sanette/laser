# ZTRACKER
#
# This file is part of the Laser library
#
# ------------------------------------------
# stand-alone tracker for demo and debugging
# ------------------------------------------
#
# (c) 2018-2022, San Vu Ngoc. University of Rennes 1.
#

import laser
import argparse
import tempfile
import timeit

def webcamTracker(camera_id):

    debug = laser.gdebug

    tmpdir = tempfile.mkdtemp(prefix='laser_')

    # tunable variable
    snake_max_size = 10
    bkgLen = 10
    sleep_time = 10 # time of inactivity in seconds before "sleep state"

    console = laser.Console("Console", (640,480), 240)
    cam, width, height = laser.openCam(camera_id)
    if isinstance(camera_id, int):
        cal = laser.calibrateLoop(cam, console)
        cal.save("%s/calibration.yml"%tmpdir)
    else:
        cal = laser.Calibration(cam)
        print ("Loading calibration data")
        cal.load(cam, "%s/calibration.yml"%os.path.dirname(camera_id))

    console.write ("------------------------------------------")
    console.write (" Press any key to start the tracking session")
    console.write ("------------------------------------------")
    console.show()
    _ = console.wait_key(0)
    
    clipBox = (1,1), (width-2, height-2) # we remove 1 pix off every border

    # Flush webcam buffer
    t0 = timeit.default_timer()
    flushSize = 10
    laser.flush(cam, flushSize)
    laser.printd ("Avg FPS for reading cam=" + str(flushSize/(timeit.default_timer()-t0)))
        
    background = laser.Background(bkgLen)
    # The background is the image that will be substracted to the current image
    # in order to detect laser motion.
    
    snake = laser.Snake(snake_max_size)
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
        img = laser.readCam(cam)
        snake.next_frame()
        show = img.copy() # this is the console image where we can draw.

        # THIS IS THE MAIN DETECTION STEP:
        t0 = timeit.default_timer()
        mask, _ = laser.oneStepTracker(background, img, show, clipBox, snake, cal)
        tt0 = laser.print_time("One Step", t0)

        if not snake.empty ():
            point = snake.last ()
            print (str(point))

        if mask != []:
            laser.printd ("There is some mask")
            #cv2.imshow("Laser", mask)

        # We display the active status in the console.
        console.reset()
        if debug:
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
        tt1 = laser.print_time("total", t1)

        # Compute FPS
        frame_count += 1
        fps = frame_count/(timeit.default_timer() - startFPS)
        laser.printd ("FPS = %f"%fps)
        if frame_count == 1000:
            frame_count = 0
            startFPS = timeit.default_timer()

        if debug:
            time_profile.append([tt0,tt1,fps])   

        # Display console and wait for key.
        idle_time = (snake.current_frame - snake.last_active_frame)/fps
        dt = (250 if idle_time > sleep_time else 50 if snake.size == 0 else 10)
        key = console.wait_key(dt)
        #time.sleep(dt/1000.)
        if key == ord('q'):
            break
        if key == ord('d'):
            debug = not debug
            laser.set_debug(debug)
            if not debug:
                background.close_window()
        if key == ord('s'):
            save = not save
            if save:
                startSave = snake.current_frame
        if key == ord('p'):
            console.write ("PAUSED. Press any key to resume")
            console.show()
            _ = console.wait_key(0)
        
    console.close()
    print ("---- Data saved in %s"%tmpdir)
    if debug:
        laser.plot_times(time_profile)


if __name__ == "__main__":
    print ("Welcome to Ztracker and the Laser library by San Vu Ngoc, University of Rennes 1.")
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
    laser.set_debug(debug)
    webcamTracker (camera_id)
    print ("Bye")
