# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:03:09 2022

@author: jonat
"""
# https://github.com/stgstg27/Background-Subtraction/blob/master/background_Subtraction.ipynb
import cv2
import numpy as np

print (cv2.__version__)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../test_video/youtube/720p/HU_Xuwei.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbgBayesianSegmentation = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
  ret, frame = cap.read()
  fgmask = fgbg.apply(frame)
  fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussain.apply(frame)
  fgbgBayesianSegmentationmask = fgbgBayesianSegmentation.apply(frame)
  fgbgBayesianSegmentationmask = cv2.morphologyEx(fgbgBayesianSegmentationmask,cv2.MORPH_OPEN,kernel)
  
  cv2.namedWindow('Background Subtraction Bayesian Segmentation',0)
  cv2.namedWindow('Background Subtraction',0)
  cv2.namedWindow('Background Subtraction Adaptive Gaussian',0)
  cv2.namedWindow('Original',0)

  cv2.resizeWindow('Original', 300,300)
  cv2.imshow('Background Subtraction Bayesian Segmentation',fgbgBayesianSegmentationmask)
  cv2.imshow('Background Subtraction',fgmask)
  cv2.imshow('Background Subtraction Adaptive Gaussian',fgbgAdaptiveGaussainmask)
  cv2.imshow('Original',frame)
  
  k = cv2.waitKey(1) & 0xff
  
  if k==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
print ('Program Closed')