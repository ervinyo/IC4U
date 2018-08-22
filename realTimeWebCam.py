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
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("3dcnnresult/3dcnnmodel.hd5")
    print("Loaded model from disk")

    loaded_model.compile(loss=categorical_crossentropy,
                         optimizer='rmsprop', metrics=['accuracy'])

    #setup cv face detection
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # setup Realsense
    # Configure depth and color streams

    # for text purpose
    txt = "OpenCV"
    txtLoad = "["
    txtDelay = "["
    txtRecord = "Capture"
    txtDel = "Delay"
    txtProbability = "0%"
    font = cv2.FONT_HERSHEY_SIMPLEX

    framearray = []
    ctrDelay = 0
    channel = 1
    gestCatch = False
    gestStart = False

    x,y,w,h = 0,0,0,0

    vc = cv2.VideoCapture(0)
    rval , firstFrame = vc.read()
    print(firstFrame.shape)
    heightc, widthc, depthcol = firstFrame.shape

    imgTxt = np.zeros((heightc, 400, 3), np.uint8)

    print(widthc)
    print(heightc)


    try:
        while True:

            dataImg = []

            rval, color_image = vc.read()
            draw_image = color_image

            #face detection here
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 2)

            if len(faces) > 0 and gestCatch == False:
                gestStart = True
                x, y, w, h = faces[0]
            else:
                x, y, w, h = x, y, w, h

            fArea = w*h
            #print(fArea)

            #if fArea > 3000:

            # crop the face then pass to resize
            cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            midx = int(x + (w * 0.5))
            midy = int(y + (h * 0.5))

            xUp = (x - (w * 3))
            yUp = (y - (h * 1.5))

            xDn = ((x + w) + (w * 1))
            yDn = ((y + h) + (h * 2))

            if xUp < 1: xUp = 0
            if xDn >= widthc: xDn = widthc

            if yUp < 1: yUp = 0
            if yDn >= heightc: yDn = heightc

            cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 0, 255), 2)

            cv2.circle(draw_image, (midx.__int__(), midy.__int__()), 10, (255, 0, 0))
            roi_color = color_image[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]


            if delayBol == False and gestStart == True:

                if depthFrame < maxFrames:
                    frame = cv2.resize(roi_color, (img_rows, img_cols))
                    framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    depthFrame = depthFrame+1
                    txtLoad = txtLoad+"["

                    gestCatch = True

                    #print(depthFrame)


                if depthFrame == maxFrames:
                    dataImg.append(framearray)
                    xx = np.array(dataImg).transpose((0, 2, 3, 1))
                    X = xx.reshape((xx.shape[0], img_rows, img_cols, maxFrames, channel))
                    X = X.astype('float32')
                    print('X_shape:{}'.format(X.shape))


                    #do prediction
                    resc = loaded_model.predict_classes(X)[0]
                    res = loaded_model.predict_proba(X)[0]

                    resultC = classGest[resc]
                    print("X=%s, Probability=%s" % (resultC, res[resc]*100))

                    for r in range(0,8):
                        print("prob: " + str(res[r]*100))

                    #show text
                    imgTxt = np.zeros((480, 400, 3), np.uint8)
                    txt = "Gesture-" + str(resultC)
                    txtProbability = str(res[resc]*100)+"%"

                    framearray = []
                    #dataImg = []
                    txtLoad = ""
                    depthFrame = 0

                    gestCatch = False
                    delayBol = True

                cv2.putText(imgTxt, txtLoad, (10, 20), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txtRecord, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txt, (10, 200), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txtProbability, (10, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            #print(delayBol)
            if delayBol == True:
                ctrDelay = ctrDelay+1
                txtDelay = txtDelay + "["
                #txtDel = "Delay"
                cv2.putText(imgTxt, txtDelay, (10, 70), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txtDel, (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if ctrDelay == delayGest:
                    ctrDelay = 0
                    txtDelay = ""
                    delayBol = False

            # Stack both images horizontally
            images = np.hstack((draw_image, imgTxt))

            # put the text here

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:

        #Stop streaming
        #pipeline.stop()
        vc.release()


if __name__ == '__main__':
        main()
