'''
File: objection_detection.py
Project: objection_detection
File Created: 19th August 2022
Author: 許迪威
'''

import cv2

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 影像侵蝕 (影像侵蝕對於移除影像中的小白雜點很有幫助，可用來去噪，例如影像中的小雜點，雜訊。)
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # 影像膨脹
#( Dilation 影像膨脹通常是配合著影像侵蝕 Erosion 使用，先使用侵蝕的方式使影像中的線條變窄，同時也去除雜訊，之後再透過 Dilation 將影像膨脹回來。)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../test_video/youtube/720p/DEURLOO_Bart.mp4')
success, frame = cap.read()
while success:

  fg_mask = bg_subtractor.apply(frame)
  # 对原始帧膨胀去噪，
  _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
  #前景区域形态学处理
  cv2.erode(thresh, erode_kernel, thresh, iterations=2)
  cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

  #绘制前景图像的检测框
  if OPENCV_MAJOR_VERSION >= 4:
      # OpenCV 4 or a later version is being used.
      contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
  else:
      # OpenCV 3 or an earlier version is being used.
      # cv2.findContours has an extra return value.
      # The extra return value is the thresholded image, which is
      # unchanged, so we can ignore it.
      _, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

  for c in contours:
      #对轮廓设置最小区域，筛选掉噪点框
      if cv2.contourArea(c) > 1000:
          #获取矩形框边界坐标
          x, y, w, h = cv2.boundingRect(c)
          cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)


  cv2.imshow('mog', fg_mask)
  cv2.imshow('thresh', thresh)
  cv2.imshow('detection', frame)

  k = cv2.waitKey(30)
  if k == 27:  # Escape
      break

  success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()