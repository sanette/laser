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

def webcamTracker(camera_id):

    app = laser.sample_app
    laser.main_loop(camera_id, app)


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
