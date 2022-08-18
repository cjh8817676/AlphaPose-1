# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:49:46 2022

@author: jonat
"""

import numpy as np
import cv2 as cv
print('here')
cap = cv.VideoCapture('../test_video/cat_jump.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()