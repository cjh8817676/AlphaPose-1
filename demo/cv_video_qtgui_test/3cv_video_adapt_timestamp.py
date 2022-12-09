# Import necessary libraries
import cv2
import os
from datetime import datetime

os.chdir(str(os.path.dirname(__file__)))
# i variable is to give unique name to images
i = 1
 
wait = 0
 
# Open the camera
video = cv2.VideoCapture('./IMG_6803-72060p.mp4')
 
 
while True:
    # Read video by read() function and it
    # will extract and  return the frame
    ret, img = video.read()
 
    # Put current DateTime on each frame
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, str(datetime.now()), (20, 40),
                font, 2, (255, 255, 255), 2, cv2.LINE_AA)
 
    # Display the image
    cv2.imshow('live video', img)
 
    # wait for user to press any key
    key = cv2.waitKey(1)
 
    # wait variable is to calculate waiting time
    # wait = wait+100
 
    if key == ord('q'):
        break
    # when it reaches to 5000 milliseconds
    # we will save that frame in given folder

# close the camera
video.release()
 
# close open windows
cv2.destroyAllWindows()