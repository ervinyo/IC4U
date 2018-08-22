from flask import Flask, jsonify, render_template, request
import traceback
from flask_restful import Resource, Api
import speech_recognition as sr
import webbrowser as wb
import time
import os
import facebook
import matplotlib
matplotlib.use('AGG')

from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from keras import backend as K
#import videoto3d

import numpy as np
import pyrealsense2 as rs
import cv2


app = Flask(__name__)
api = Api(app)


''' Page '''

########              Main              ############
@app.route('/')
def index():
    return render_template('home.html')

########              Yotube              ############
idyoutube=''

@app.route('/youtube/')
def youtube():
    global idyoutube
    print("-------------------------------------------")
    print(idyoutube)
    return render_template('youtube.html', id = idyoutube)	

@app.route('/listvideo/')
def listvideo():
    return render_template('listvideo.html')	
@app.route('/setyoutubeID', methods=['GET', 'POST'])
def setyoutubeID():
    global idyoutube
    if request.method == "POST":
        #return ("22")
        idyoutube =  request.form['idplay']
        #idyoutube =  request.form.get('idplay')
        return str(idyoutube)
    else:
        return "error"

########              Facebook              ############
graph = facebook.GraphAPI(access_token="EAAegm6LP18oBAFVwezZBWISFve98d6R6oJR4jpLAV3sU3XzAnqPg8vPTqbxpf3St7XawgfQVHMZAMvXIVqvlySjO76biYgdW3hFZASCX6SJ1ZCyHyWO4kLY06jSj5ZAdPKc8dMpCEZAGoZChKvmdRx5alGgs2GxCBcZD", version='2.7')

idphoto=''

@app.route('/photo/')
def photo():
    global idphoto
    print("-------------------------------------------")
    print(idphoto)
    return render_template('photo.html', id = idphoto)	
	
@app.route('/listphoto')
def listphoto():
    return render_template('listphoto.html')

@app.route('/biodata')
def biodata():
    return render_template('biodata.html')
	
@app.route('/listpost')
def post():
    return render_template('listpost.html')
		
@app.route('/getPhoto', methods=['GET', 'POST'])
def getPhoto():
    #id = graph.request("/me?fields=id")
    #name = graph.request("/me?fields=name")
    #photo = graph.request("/me?fields=photos{images}")
    photo = graph.request("me?fields=albums{photos{images}}")
    #print(photo)
    photo = jsonify(photo)
    return photo

@app.route('/getBiodata', methods=['GET', 'POST'])
def getBiodata():
    #id = graph.request("/me?fields=id")
    #name = graph.request("/me?fields=name")
    #photo = graph.request("/me?fields=photos{images}")
    bio = graph.request("me?fields=about,birthday,address,age_range,education,gender,hometown,name,relationship_status,email")
    #print(photo)
    bio = jsonify(bio)
    return bio

@app.route('/getPost', methods=['GET', 'POST'])
def getPost():
    post = graph.request("me?fields=feed{message}")
    post = jsonify(post)
    return post
	
@app.route('/fb')
def fb():
    id = graph.request("/me?fields=id")
    name = graph.request("/me?fields=name")
    email = graph.request("/me?fields=email")
    gender = graph.request("/me?fields=gender")
    photo = graph.request("/me?fields=photos{images}")
    post = graph.request("me?fields=feed{message}")
    return render_template('display2.html', id = id, name=name, email=email, gender=gender, photo=photo, post=post)
	
@app.route('/facebook')
def facebook():
    return render_template('facebook.html')
		
''' Command '''

#######################################################
########              Voice              ############
#######################################################
@app.route('/voice', methods=['GET', 'POST'])
def voice():
    r = sr.Recognizer() 
    with sr.Microphone() as source:
        f = open ('result.txt','w')
        f.write("wait")
        #print ("Please wait a moment...  Calibrating microphone NOW~")
        # listen for 1 seconds and create the ambient noise energy level 
        r.adjust_for_ambient_noise (source, duration=2) 
        #print ("Now, please say something !!!")
        res = 'tes'    
        audio = r.listen (source)
    try:
        #f = open ('result.txt','w')
        text = r.recognize_google(audio, language="EN")
        if text == 'close':
            return "close"
        res = (text)
        
        #print (r.recognize_google(audio, language="EN"), file = f)
        #f_text =  text + ".com"
        #wb.get (chrome_path) .open (f_text)
        #wb.open('http://127.0.0.1:5000/' + f_text)
        #print ("What you said has been saved as [result.txt] :)")
        return res
    except sr.UnknownValueError:
        return ("0")
    except sr.RequestError as e:
        return ("0")
    #f.close()
#######################################################
########              Gesture              ############
#######################################################
#######################################################
#Global Variable
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
handin = False
handx = 0
handy = 0
facex = 0
facey = 0

@app.route('/cekHand', methods=['GET', 'POST'])	
def cekHand():
    global handin 
    global handx
    global handy
    #return str(handy)
    aa="out"
    if(handin==True):
        aa="in"
    else:
        aa="out"
    #return jsonify({0:aa, 1:handx, 2:handy})
    return str(handin)

def cekState(state,gest):
    classGest = ['1','11','12','13','4','5','7','8']
    if(state==1):
        gestures={11,12,13,5,7}
    elif(state==2):
        gestures={11,12,13,5,7}
    gestmax = 0
    indexmax = 0
    for i in gestures:
        index = classGest.index(str(i))
        print(str(index)+"--"+str(gest[index])+"--"+str(gestmax))
        if gest[index]>gestmax:
            gestmax = gest[index]
            indexmax = index
        print("indexmax="+str(indexmax))
    res	= indexmax
    print(res)
    return res

def checkhandIn(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image):

    global handin
    global handx
    global handy
    if boxCounter == True:

        #dRefMaxF = deptRef - 0.15
        dRefMaxF = deptRef - 0.2
        # dRefMinF = dRefVal - 0.1

        boxXup = int(midx - (w * 3))
        boxYup = int(midy - (h * 1.5))

        boxXdn = int(midx + (w * 1))
        boxYdn = int(midy + (h * 2))

        if boxXup < 1: boxXup = 0
        if boxXdn >= 848: boxXdn = 848

        if boxYup < 1: boxYup = 0
        if boxYdn >= 480: boxYdn = 480
        #for x in range(boxYup, boxYdn) for y in range(boxXup, boxXdn) if (x * 80) + (z * 65) + (y * 50) == 1950
        ii = ([j,i] for j in range(boxYup, boxYdn) for i in range(boxXup, boxXdn) if depth_imageS.item(j, i) > dRefMaxF or depth_imageS.item(j, i) == 0)
        for j,i in ii:
            boxColor.itemset((j, i, 0), 0)
            boxColor.itemset((j, i, 1), 0)
            boxColor.itemset((j, i, 2), 0)
        '''
        ii = ([j,i] for j in range(boxYup, boxYdn) for i in range(boxXup, boxXdn) if depth_imageS.item(j, i) > dRefMaxF or depth_imageS.item(j, i) == 0)
        for j,i in ii:
            boxColor.itemset((j, i, 0), 0)
            boxColor.itemset((j, i, 1), 0)
            boxColor.itemset((j, i, 2), 0)

        for j in range(boxYup, boxYdn):
            for i in range(boxXup, boxXdn):
                if depth_imageS.item(j, i) > dRefMaxF or depth_imageS.item(j, i) == 0:
                    boxColor.itemset((j, i, 0), 0)
                    boxColor.itemset((j, i, 1), 0)
                    boxColor.itemset((j, i, 2), 0)
        '''
        roi_boxCounter = boxColor[boxYup.__int__():boxYdn.__int__(), boxXup.__int__():boxXdn.__int__()]
        #roi_boxCounter = cv2.resize(roi_boxCounter,(32,32))
        graybox = cv2.cvtColor(roi_boxCounter, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(graybox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
        #idx = 0
        '''
        if(len(contours)>0):
            cnt = max(contours, key = cv2.contourArea)
            xb, yb, wb, hb = cv2.boundingRect(cnt)
            areaCnt = wb * hb
            if areaCnt > 500:
                # print(areaCnt)
                handin = True
        '''
        for cnt in contours:
            #idx += 1
            xb, yb, wb, hb = cv2.boundingRect(cnt)
            areaCnt = wb * hb
            #print(areaCnt)
            if areaCnt > 500:
                # print(areaCnt)
                handin = True
                break

    '''
        if handin==True:
            cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                      (0, 255, 255), 2)
        else:
            cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                          (0, 0, 255), 2)
    '''''
    return handin

@app.route('/gesture', methods=['GET', 'POST'])
def gesture():
    #return('gesture')
    #global testDir
    #global img_rows, img_cols, maxFrames
    #global depthFrame
#crop parameter
    #global xupam
    #global yupam
    #global xdpam
    #global ydpam
    #global classGest
    #global delayGest
    #global delayBol
#load the model and weight
    #global json_file
    #global loaded_model_json
    #global loaded_model
#setup cv face detection
    #global face_cascade
# setup Realsense
# Configure depth and color streams
# for text purpose
    global handin
    global facex
    global facey
    global txt
    global txtLoad
    global txtDelay
    global txtRecord
    global txtDel
    global txtProbability
    global font
    #global framearray
    #global ctrDelay
    #global channel
    #global gestCatch
    #global gestStart
    #global x,y,w,h
    #global vc
    #global rval , firstFrame
    #global heightc, widthc, depthcol
    #global imgTxt
    #global resultC
    #global count
    #global stat
    pipeline = rs.pipeline()
    K.clear_session()
    testDir = "videoTest/"
    img_rows, img_cols, maxFrames = 32, 32, 55
    depthFrame = 0
#crop parameter
    xupam = 350
    yupam = 200

    xdpam = 250
    ydpam = 300
    depthFrame = 0
    cameraHeightR = 480
    #cameraWidthR = 848
    cameraWidthR = 848
    frameRateR = 60
    classGest = ['1','11','12','13','4','5','7','8']
    delayGest = 20
    delayBol = False
    framearray = []
    ctrDelay = 0
    channel = 1
    gestCatch = False
    gestStart = False
    backgroundRemove = True
    x,y,w,h = 0,0,0,0
    count=0
    boxCounter = True
#load the model and weight
    json_file = open('3dcnnresult/24/3dcnnmodel.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
	# load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    #loaded_model.load_weights("3dcnnresult/3dcnnmodel.hd5")
    loaded_model.load_weights("3dcnnresult/24/3dcnnmodel.hd5")
#setup cv face detection
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    
    config = rs.config()
    config.enable_stream(rs.stream.depth, cameraWidthR, cameraHeightR, rs.format.z16, frameRateR)
    config.enable_stream(rs.stream.color, cameraWidthR, cameraHeightR, rs.format.bgr8, frameRateR)
    '''vc = cv2.VideoCapture(0)
    rval , firstFrame = vc.read()
    heightc, widthc, depthcol = firstFrame.shape
    imgTxt = np.zeros((heightc, 400, 3), np.uint8)
    #print('tryyyyy1')'''
    stat =0
    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print "Depth Scale is: " , depth_scale

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2 # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    #print("tes====================")
    # for text purpose
    imgTxt = np.zeros((480, 400, 3), np.uint8)
    txt = "OpenCV"
    txtLoad = "["
    txtDelay = "["
    txtRecord = "Capture"
    txtDel = "Delay"
    txtProbability = "0%"
    font = cv2.FONT_HERSHEY_SIMPLEX

    align_to = rs.stream.color
    align = rs.align(align_to)
#print("Loaded model from disk")
    count=0
    loaded_model.compile(loss=categorical_crossentropy,
                     optimizer='rmsprop', metrics=['accuracy'])
    while True:
        if True:
            dataImg = []

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if (backgroundRemove == True):
                # Remove background - Set pixels further than clipping_distance to grey
                # grey_color = 153
                grey_color = 0
                depth_image_3d = np.dstack(
                    (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color,
                                      color_image)

                color_image = bg_removed
                draw_image = color_image

            else:
                draw_image = color_image

                        #face detection here
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray', gray)
            #cv2.waitKey(5000)

            faces = face_cascade.detectMultiScale(gray, 1.1, 2)

            if len(faces) > 0:
                if gestCatch == False and delayBol == False:

                    for f in faces:
                        xf, yf, wf, hf = f
                        farea = wf*hf
                        #print(farea)
                        if farea > 9000 and farea < 12000:
                            x, y, w, h = f
                            #gestStart = True
                            #x, y, w, h = faces[0]
            else:
                x, y, w, h = x, y, w, h

            fArea = w*h
            
            '''
            midx = int(x + (w * 0.5))
            midy = int(y + (h * 0.5))

            xUp = (x - (w * 3))
            yUp = (y - (h * 1.5))

            xDn = ((x + w) + (w * 1))
            yDn = ((y + h) + (h * 2))

            if xUp < 1: xUp = 0
            if xDn >= cameraWidthR: xDn = cameraWidthR

            if yUp < 1: yUp = 0
            if yDn >= cameraHeightR: yDn = cameraHeightR

            if handin == False:
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 0, 255),
                              2)
            else:
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 255, 0),
                              2)
            cv2.circle(draw_image, (midx.__int__(), midy.__int__()), 10, (255, 0, 0))

            # region of interest
            roi_gray = gray[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]
            '''
            midx = int(x + (w * 0.5))
            midy = int(y + (h * 0.5))

            xUp = (x - (w * 3))
            yUp = (y - (h * 1.5))

            xDn = ((x + w) + (w * 1))
            yDn = ((y + h) + (h * 2))

            if xUp < 1: xUp = 0
            if xDn >= cameraWidthR: xDn = cameraWidthR

            if yUp < 1: yUp = 0
            if yDn >= cameraHeightR: yDn = cameraHeightR

            if handin == False:
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 0, 255), 2)
            else:
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 255, 0),
                              2)
            cv2.circle(draw_image, (midx.__int__(), midy.__int__()), 10, (255, 0, 0))

            roi_color = color_image[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]

            #find the depth of middle point of face
            if backgroundRemove == True and gestCatch == False:
                #e1 = cv2.getTickCount()
                depth_imageS = depth_image * depth_scale
                deptRef = depth_imageS.item(midy, midx)
                # print(clipping_distance)

                clipping_distance = (deptRef + 0.2) / depth_scale
                boxColor = color_image.copy()

                handin = checkhandIn(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image)

                e2 = cv2.getTickCount()

                #times = (e2 - e1) / cv2.getTickFrequency()
                #print(times)


                if handin == True:
                    gestStart = True
                else:
                    gestStart = False

            if delayBol == False and gestStart == True:

                if depthFrame < maxFrames:
                    frame = cv2.resize(roi_color, (img_rows, img_cols))
                    framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    depthFrame = depthFrame+1
                    txtLoad = txtLoad+"["
                    count=count+1
                    gestCatch = True

                    #print(depthFrame)


                if depthFrame == maxFrames:
                    dataImg.append(framearray)
                    xx = np.array(dataImg).transpose((0, 2, 3, 1))
                    X = xx.reshape((xx.shape[0], img_rows, img_cols, maxFrames, channel))
                    X = X.astype('float32')
                    #print('X_shape:{}'.format(X.shape))


                    #do prediction
                    resc = loaded_model.predict_classes(X)[0]
                    res = loaded_model.predict_proba(X)[0]

                    resultC = classGest[resc]
                    #print("X=%s, Probability=%s" % (resultC, res[resc]*100))

                    for r in range(0,8):
                        print("prob: " + str(res[r]*100))

                    #show text
                    #imgTxt = np.zeros((480, 400, 3), np.uint8)
                    txt = "Gesture-" + str(resultC)
                    txtProbability = str(res[resc]*100)+"%"

                    framearray = []
                    #dataImg = []
                    txtLoad = ""
                    depthFrame = 0
                    handin = False
                    gestCatch = False
                    delayBol = True

                #cv2.putText(imgTxt, txtLoad, (10, 20), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                #.putText(imgTxt, txtRecord, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgTxt, txt, (10, 200), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgTxt, txtProbability, (10, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            #print(delayBol)
            if delayBol == True:
                ctrDelay = ctrDelay+1
                txtDelay = txtDelay + "["
                #txtDel = "Delay"
                #cv2.putText(imgTxt, txtDelay, (10, 70), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgTxt, txtDel, (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if ctrDelay == delayGest:
                    ctrDelay = 0
                    txtDelay = ""
                    delayBol = False

            # Stack both images horizontally
            #images = np.hstack((draw_image, imgTxt))

            # put the text here

            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', images)
            #cv2.waitKey(1)
            
            if count==maxFrames:
                #vc.release()
                K.clear_session()
                pipeline.stop()
                return resultC
        '''
        except ValueError:
            vc.release()
            err = str(ValueError)
            print(err)
            return(err)
        finally:
            vc.release()
            return(err)
			'''
        #return resultC
        #Stop streaming
        #pipeline.stop()
        #vc.release()
		
    
#######################################################


if __name__ == '__main__':
    #gesture()
    #vc = cv2.VideoCapture(0)
    app.run(debug = True)

