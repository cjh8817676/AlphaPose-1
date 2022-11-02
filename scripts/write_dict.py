# -*- coding: utf-8 -*-
import cv2
import json

#  該程式只針對有進行過--pose_track的影片產生出的json。
if __name__ == "__main__":

    # video_path = '../test_video/test_video/IMG_6805.MOV'
    video_path = '../test_video/NTSU/IMG_6803-72060p.mp4'
    # video_path = '../test_video/youtube/720p/KARIMI_Milad.mp4'
    
    file_name = video_path.split('/')[-1]
    
    stream = cv2.VideoCapture(video_path)  # 影像路徑
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")              # Ubuntu 20.04 fourcc
    # fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
    fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) # 影片寬
    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影片長
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': (h,w)} # 影片資訊
    
    out = cv2.VideoWriter(file_name, fourcc, fps, (w,  h))
    
    json_data = []
    files_dir = '../examples/res/'+ file_name + '.json'   # json 路徑
    with open(files_dir) as f:
        content = json.load(f)
            
    img_id = 0
    json_img_id = 0
    while stream.isOpened():
        ret, frame = stream.read()                                # frame : (origin_w,origin_h,3)的 Array
        if not ret or img_id >= (datalen-1) or json_img_id >= (len(content)-1):
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        
        while (img_id == int(content[json_img_id]['image_id'].split('.')[0])):
            bbox = content[json_img_id]['box']
            k = int(content[json_img_id]['idx'])
            frame =  cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0, 255, 0))
            frame = cv2.putText(frame, str(k), (int(bbox[0]),int(bbox[1])) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
            json_img_id+=1
        
        img_id += 1
    
        
        out.write(frame)
        frame = cv2.resize(frame, (1600,900)) 
        cv2.imshow('detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    out.release()
    stream.release()
    cv2.destroyAllWindows()
        