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


id_list = []   # 紀錄已出現過的box id，不同id有可能是同類。
bbox_list = [] # 記錄各個box id在每一幀的位置 (前500幀就決定所有的獨一無二的種類)
mark_move = [] # 
FRAME_NUM = 30 
DETECT_MOVE_FRAME = 500 #前500幀 裡所有出現的bbox編號都視為獨一無二的存在。  (最好選體操選手碰到單槓時且出現bbox 當下的幀)
MIN_BOX_SIZE = (80*(h/720) ,80 * (w/1280) )
MAX_BOX_SIZE = (1000*(h/720) ,1000* (w/1280))
MOVE_THRESHOLD = 50* ((h/720)*(w/1280))
STATE_FALSE   = 0
STATE_TRUE    = 1
STATE_TOO_MAX = 2
STATE_TOO_MIN = 3


def get_max_pos( idx , frame_id , bbox ): #尋找bbox的位移量
    bboxs = bbox_list[ idx ]
    max_pos = 0
    for i in bboxs.keys():               # 不同bbox_id 但視為同類
        bbox_old = bboxs[i]
        if (i < frame_id) and ( i >= (frame_id - FRAME_NUM) ):  # 計算30幀內的位移量  (60fps，所以是0.5秒看有沒有移動100個pixel面積)
            pos = abs( bbox_old[0] - bbox[0] ) + abs( bbox_old[1] - bbox[1] )
            if max_pos < pos:
                max_pos = pos
    return max_pos            

'''
ra = Rectangle(3., 3., 5., 5.)
rb = Rectangle(1., 1., 4., 3.5)
# intersection here is (3, 3, 4, 3.5), or an area of 1*.5=.5

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

'''

def get_is_cover( idx , bbox ): # 確認是否重疊
    bboxs = bbox_list[ idx ]
    for i in bboxs.keys():  # 某個id的 bbox 在每一幀的座標
        bbox_old = bboxs[i]
        dx = min(bbox_old[0]+bbox_old[2], bbox[0]+bbox[2] ) - max(bbox_old[0], bbox[0] )
        dy = min(bbox_old[1]+bbox_old[3], bbox[1]+bbox[3] ) - max(bbox_old[1], bbox[1] )
        if (dx>=0) and (dy>=0):
            return dx*dy
        
def put_id(id,bbox,frame_id):            # 將box id 丟入 id_list，並查看有沒有重複出現過
    print(frame_id)
    idx = -1
    for x,ids in enumerate( id_list ):   # 有重複
        if id in ids:
            idx = x                      # 重複的bbox的id
            break
    if idx >= 0:                         # 有重複，就只記錄在該幀的位置。  
        ids = id_list[idx]
        bbox_list[idx][frame_id] = bbox 
        pos = get_max_pos( idx , frame_id , bbox )  # 計算移動
        if pos > MOVE_THRESHOLD and frame_id < DETECT_MOVE_FRAME : # 有重複出現，才有辦法判斷是否移動
            print(f"發現ID: {id} 移動: {pos}")
            mark_move[idx] = STATE_TRUE  # 有動很大的人，才會被設定成true。
    else:                                # 沒重複
        if frame_id < DETECT_MOVE_FRAME: # 前 500 幀決定存在的bbox id。 在這裡姑且"新出現的"都視為獨一無二的存在
            print("出現新ID:", id)
            id_list.append( [id] )
            bbox_list.append( { frame_id : bbox } )
            mark_move.append(STATE_FALSE) # 剛出現的沒有移動
        else:                           # 500幀後 "新出現的"，是因為Human Re-id 的錯誤。 利用bbox的比較去找尋前500幀的同類
            print("新出現的:", id)
            for idx,i in enumerate(mark_move):     # 一定是移動中的bbox才有可能導致bbox的id突變。 
                if i and get_is_cover(idx , bbox): #check 是否已經存在同類
                    id_list[idx].append(id)
                    bbox_list[idx][frame_id] = bbox
                        
def is_show(id):                          # 只有移動中的bbox的才會被show出
    idx = -1
    for x,ids in enumerate( id_list ):
        if id in ids:
            idx = x
            break
    return mark_move[idx] == STATE_TRUE
    
def get_id(id):                           # 找尋根源同類
    idx = -1
    for x,ids in enumerate( id_list ):
        if id in ids:
            return x
    
def hidden_max_min():                    # 太大太小的bbox去掉
    for idx in range( len(id_list) ):
        num = 0
        size_x = 0
        size_y = 0
        id = id_list[idx]
        for frame_id in bbox_list[idx].keys():    # 某個bbox 存在的每一幀時的大小。  
            num += 1
            size_x += bbox_list[idx][frame_id][2] # w
            size_y += bbox_list[idx][frame_id][3] # h　
        size_x /= num
        size_y /= num
        print(f"ID:{id} X:{size_x} , Y:{size_y}")
        if size_x < MIN_BOX_SIZE[0] and size_y < MIN_BOX_SIZE[1]:
            mark_move[idx] = STATE_TOO_MIN
        elif size_x > MAX_BOX_SIZE[0] and size_y > MAX_BOX_SIZE[1]:
            mark_move[idx] = STATE_TOO_MAX

if __name__ == "__main__":
    
    json_data = []
    files_dir = '../examples/res/'+ file_name + '.json'  # json 路徑 。   這裡的json裡的img_id與idx都已經經過排序。
    with open(files_dir) as f:
        content = json.load(f)
    
    json_img_id = 0
    img_id = 0
    while json_img_id < len(content):
        while json_img_id < len(content) and (img_id == int(content[json_img_id]['image_id'].split('.')[0])):
            bbox = content[json_img_id]['box']
            k = int(content[json_img_id]['idx']) # Human Re-id 的編號
            put_id(k , bbox, img_id)
            json_img_id+=1
        img_id += 1
    
    hidden_max_min()
    
    json_data = []
    with open(files_dir) as f:
        content2 = json.load(f)
            
    img_id = 0
    json_img_id = 0
    
    img_id2= 0
    json_img_id2 = 0
    
    out = cv2.VideoWriter(file_name, fourcc, fps, (w,  h))
    
    
    while stream.isOpened():
        ret, frame = stream.read()                                # frame : (origin_w,origin_h,3)的 Array
        frame2 = frame * 1
        if not ret or img_id >= (datalen-1):
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if json_img_id < (len(content)-1):
            while (img_id == int(content[json_img_id]['image_id'].split('.')[0])): # 取得一幀內所有的bbox
                bbox = content[json_img_id]['box']
                k = int(content[json_img_id]['idx'])
                if is_show(k) and bbox[2] >= MIN_BOX_SIZE[0] and bbox[3] >= MIN_BOX_SIZE[1]: # 符合規則的才會被顯現出來。
                    k = get_id(k)
                    frame =  cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0, 255, 0))
                    frame = cv2.putText(frame, str(k), (int(bbox[0]),int(bbox[1])) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
                json_img_id+=1
                
            while (img_id2 == int(content2[json_img_id2]['image_id'].split('.')[0])):
                bbox = content2[json_img_id2]['box']
                k = int(content2[json_img_id2]['idx'])
                frame2 =  cv2.rectangle(frame2, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0, 255, 0))
                frame2 = cv2.putText(frame2, str(k), (int(bbox[0]),int(bbox[1])) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
                json_img_id2+=1
                
            img_id2 += 1
            img_id += 1
            out.write(frame)
            
            numpy_horizontal_concat = np.concatenate((frame2, frame), axis=1)
            cv2.imshow('detection', numpy_horizontal_concat)
            
            imS = cv2.resize(numpy_horizontal_concat, (1800,600))
            cv2.imshow('detection', imS)
        else:
            print('sth miss')
            img_id += 1
            out.write(frame)
            cv2.imshow('detection', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
       
    out.release()
    stream.release()
    cv2.destroyAllWindows()
        

        