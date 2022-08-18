# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:43:57 2022

@author: jonat
"""


# Python code for Background subtraction using OpenCV
import numpy as np
import cv2
 
cap = cv2.VideoCapture('../test_video/cat_jump.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
 
while(1):
    ret, frame = cap.read()
 
    fgmask = fgbg.apply(frame)
  
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('frame',frame )
 
     
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
     
 
cap.release()
cv2.destroyAllWindows()