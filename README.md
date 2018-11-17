# laser
Laser pointer tracker

**laser** is a small library to detect the light beam of a laser pointer using a plain webcam.

It also works with other types of focused lights (some flashlights can adjust the size of the beam)

It includes a calibration utility and a demo program.

To launch the demo with webcam device #1:
```python ./laser.py -camera 1```

## Requirements

python 2 and opencv 2.4 (these are the default packages in ubuntu 16)
```
  sudo apt install python-opencv
  sudo apt install python-yaml
```
