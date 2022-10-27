# -*- coding: utf-8 -*-

import cv2
import json
import time


id_list = []   # 紀錄已出現過的box id (按照bbox id順序) 從1開始
bbox_list = [] # 記錄各個box id在每一幀的位置 (按照bbox id 順序)
mark_move = []
MOVE_THRESHOLD = 100
STATE_FALSE   = 0
STATE_TRUE    = 1

def mat_inter(box1,box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False
def solve_coincide(box1,box2):  # 回傳重和比例
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inter(box1,box2)==True:
    
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        
        intersection=col*row
        
        area1=(x02-x01)*(y02-y01)
        area2=(x12-x11)*(y12-y11)
        
        coincide=intersection/(area1+area2-intersection)
        return coincide
    else:
        return False

def compare_box_cover(last_bbox,bbox):  # box: (x,y,x+w,y+h)

    x1,y1,x2,y2=bbox[0],bbox[1],(bbox[0]+bbox[2]),(bbox[1]+bbox[3])
    x3,y3,x4,y4 = last_bbox[0],last_bbox[1],(last_bbox[0]+last_bbox[2]),(last_bbox[1]+last_bbox[3])
    
    box=(x1,y1,x2,y2)
    last_box=(x3,y3,x4,y4)
    
    if mat_inter( last_box , box ):
        return solve_coincide(last_box, box)
    else:
        return 0
    
    

def put_id(id,bbox,frame_id):            # bbox: [x,y,x+w,y+h]
    idx = -1
    for x,ids in enumerate( id_list ):   # 有重複
        if id in ids:
            idx = x                      # bbox的id
            break
    if idx >= 0:                         # 有重複，就只記錄在該幀的位置
        ids = id_list[idx]
        bbox_list[idx][frame_id] = bbox 
    else:                                # 一開始沒有重複
        if frame_id == 0:
            id_list.append( [id] )
            bbox_list.append( { frame_id : bbox } )
            mark_move.append(STATE_FALSE)
        else:                           # 沒重複，去上一幀所有bbox與其重疊多少
            compare_cover = []
            last_frame_id = frame_id - 1
            for box in bbox_list:         # 在上一幀當中所有的bbox做面積重疊比較，並找出最小值
                area = compare_box_cover(box[last_frame_id], bbox)
                compare_cover.append(abs(area))
            
            if max(compare_cover) == 0:  # 確認毫無重疊，全新無變換。
                id_list.append( [id] )
                bbox_list.append( { frame_id : bbox } )
                mark_move.append(STATE_FALSE)
            else:
                
                
                id_list.append( [id] )
                bbox_list.append( { frame_id : bbox } )
                mark_move.append(STATE_FALSE)
                

if __name__ == "__main__":
    
    video_path = '../test_video/NTSU/IMG_6803-72060p.mp4'
    # video_path = '../test_video/youtube/720p/KARIMI_Milad.mp4'
    file_name = video_path.split('/')[-1]
    
    json_data = []
    files_dir = '../examples/res/'+ file_name + '.json'  # json 路徑
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
    
    # hidden_max_min()
    
    # stream = cv2.VideoCapture(video_path)  # 影像路徑
    # datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
    # fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
    # fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
    # w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) # 影片寬
    # h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影片長
    # videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': (h,w)} # 影片資訊
    
    # json_data = []
    # with open(files_dir) as f:
    #     content = json.load(f)
            
    # img_id = 0
    # json_img_id = 0
    
    # out = cv2.VideoWriter(file_name, fourcc, fps, (w,  h))
    
    
    # while stream.isOpened():
    #     ret, frame = stream.read()                                # frame : (origin_w,origin_h,3)的 Array
    #     if not ret or img_id >= (datalen-1):
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
        
    #     if json_img_id < (len(content)-1):
    #         while (img_id == int(content[json_img_id]['image_id'].split('.')[0])):
    #             bbox = content[json_img_id]['box']
    #             k = int(content[json_img_id]['idx'])
    #             if is_show(k) and bbox[2] >= MIN_BOX_SIZE[0] and bbox[3] >= MIN_BOX_SIZE[1]:
    #                 k = get_id(k)
    #                 frame =  cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0, 255, 0))
    #                 frame = cv2.putText(frame, str(k), (int(bbox[0]),int(bbox[1])) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
    #             json_img_id+=1
    #         img_id += 1
    #         out.write(frame)
    #         cv2.imshow('detection', frame)
    #     else:
    #         img_id += 1
    #         out.write(frame)
    #         cv2.imshow('detection', frame)
        
    #     if cv2.waitKey(1) == ord('q'):
    #         break
       
    # out.release()
    # stream.release()
    # cv2.destroyAllWindows()
        

        