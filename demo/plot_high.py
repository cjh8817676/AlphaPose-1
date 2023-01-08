# -*- coding: utf-8 -*-

import cv2
import json
import time
import numpy as np

# video_path = '../test_video/test_video/IMG_6805.MOV'         # DETECT_MOVE_FRAME=350  (h=2160, w=3840) 60fps
video_path = '../test_video/NTSU/IMG_6803-72060p.mp4'      # DETECT_MOVE_FRAME=500  (h=720,w=1280) 60fps
# video_path = '../test_video/youtube/720p/KARIMI_Milad.mp4' # DETECT_MOVE_FRAME=250  (h=720,w=1280) 30fps
file_name = video_path.split('/')[-1]

stream = cv2.VideoCapture(video_path)  # 影像路徑
datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
fourcc = cv2.VideoWriter_fourcc(*"mp4v")              # Ubuntu 20.04 fourcc
# fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) # 影片寬
h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影片長
videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': (h,w)} # 影片資訊

if __name__ == "__main__":
    
    files_dir = '../examples/res/'+ file_name + '.json'  # json 路徑 。   這裡的json裡的img_id與idx都已經經過排序。
    print(files_dir)
    with open(files_dir) as f:
        content = json.load(f)
    img_id2= 0
    json_img_id2 = 0
    
    out = cv2.VideoWriter(file_name, fourcc, fps, (w,  h))
    
    while stream.isOpened():
        ret, frame = stream.read()                                # frame : (origin_w,origin_h,3)的 Array
        frame2 = frame * 1

        while (img_id2 == int(content[json_img_id2]['image_id'].split('.')[0])): # 讀取正常json
            bbox = content[json_img_id2]['box']
            k = int(content[json_img_id2]['idx'])
            frame2 =  cv2.rectangle(frame2, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0, 255, 0))
            frame2 = cv2.putText(frame2, str(k), (int(bbox[0]),int(bbox[1])) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
            json_img_id2+=1
        img_id2 += 1
        
        imS = cv2.resize(frame2, (1800,600))
        cv2.imshow('detection', imS)

        
        if cv2.waitKey(1) == ord('q'):
            break
       
    stream.release()
    cv2.destroyAllWindows()
        