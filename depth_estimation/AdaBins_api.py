from AdaBins.models import UnetAdaptiveBins
from AdaBins import model_io
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'AdaBins/'))

from infer import InferenceHelper

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80
N_BINS = 256 




if __name__ == "__main__":
    """ 
    predictions using nyu dataset
    """
    infer_helper = InferenceHelper(dataset='nyu')   # load depth_estimation model
    
    # predict depth of a single pillow image
    img = Image.open("./AdaBins/test_imgs/pic0.jpg").convert('RGB')  # any rgb pillow image
    newsize = (640, 480)
    img = img.resize(newsize)
    print('here',type(img),img.size)
    bin_centers, predicted_depth = infer_helper.predict_pil(img)
    
    plt.imshow(predicted_depth[0][0], cmap='plasma')
    plt.show()