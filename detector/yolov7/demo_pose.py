from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

if __name__ == "__main__":
    path = './test_video/cat_jump.mp4'                  # 影片路徑
    stream = cv2.VideoCapture(path)
                                     
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
    fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) # 影片寬
    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影片長
    print('fps:',fps)
    print('寬與長:',w,h)
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, (640,448))


    orig_dim_list = torch.Tensor([[w,h,w,h]])
    torch.set_grad_enabled(False)
    while stream.isOpened():
        ret, frame = stream.read()  # frame : (origin_w,origin_h,3)的 Array

        image = letterbox(frame, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(device)   
        
        output, _ = model(image)

        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        
        cc2.recan
        cv2.imshow('detection', nimg)
        writer.write(nimg)

        if cv2.waitKey(1) == ord('q'):
            break

    writer.release()
    stream.release()
    cv2.destroyAllWindows()



