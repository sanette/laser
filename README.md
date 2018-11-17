# laser
Laser pointer tracker 

**laser** is a small library to detect the light beam of a laser pointer using a plain webcam.

It also works with other types of focused lights (some flashlights can adjust the size of the beam).

It works on any kind of non-moving background (it doesn't have to be white, or even uniform).

It includes a calibration utility and a demo program.

To launch the demo with webcam device #1:
```python ./laser.py -camera 1```

![LAZER](https://github.com/sanette/laser/blob/master/lazer.png) 

The code contains extensive debugging information that will be displayed if you use the "-d" flag.
```
usage: laser.py [-h] [-d] [-c CAMERA]

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           set debug mode
  -c CAMERA, --camera CAMERA
                        set camera device id
```

## Requirements

python 2 and opencv 2.4 (these are the default packages in ubuntu 16)
```
  sudo apt install python-opencv
  sudo apt install python-yaml
```
## How it works

The detection is based on an original algorithm, which combines a 'traditional' motion detection by background substraction with a new 'smoothness detector' which accounts for the fact that the hand motion is not completely erratic.

Detecting laser dots is not a new subject. Several algorithms exist, but none of them is optimal. See for instance a review paper here: http://www.jatit.org/volumes/Vol70No2/18Vol70No2.pdf

The approach that I use is adapted to a situation were the laser dot is projected on a still background, with stable lighting conditions. Typically, it should work indoors, eg. in a seminar room when the laser is used as a pointer for a computer presentation with a videoprojector.

In further versions I will try to add more geometric conditions to make the detection even more robust.

## Calibration

There are default values that should work out-of-the-box, but for better results, I recommend running the calibration function.
