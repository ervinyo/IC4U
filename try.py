import os

import matplotlib
matplotlib.use('AGG')

from keras.models import model_from_json
from keras.losses import categorical_crossentropy
#import videoto3d

import numpy as np
import pyrealsense2 as rs
import cv2

def main():

    testDir = "videoTest/"
    img_rows, img_cols, maxFrames = 32, 32, 100
    depthFrame = 0

    #crop parameter
    xupam = 350
    yupam = 200

    xdpam = 250
    ydpam = 300

    classGest = ['1','11','12','13','4','5','7','8']
    delayGest = 20
    delayBol = False


    #load the model and weight
    json_file = open('3dcnnresult/3dcnnmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    #print(loaded_model_json)
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("3dcnnresult/3dcnnmodel.hd5")
    print("Loaded model from disk")
	
if __name__ == '__main__':
        main()