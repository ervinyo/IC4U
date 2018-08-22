from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import traceback
from threading import Lock
from flask_restful import Resource, Api
import speech_recognition as sr
import webbrowser as wb
from utils import detector_utils as detector_utils
import tensorflow as tf
import argparse
import time
import os
import facebook
import matplotlib
import csv
#matplotlib.use('AGG')
from datetime import datetime
from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from keras import backend as K
#import videoto3d

import numpy as np
import pyrealsense2 as rs
import cv2


async_mode = None
# set to True to inform that the app needs to be re-created
app = Flask(__name__)
api = Api(app)
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()
''' Page '''
handin=True
########              Main              ############
@app.route('/')
def index():
    return render_template('home.html', async_mode=socketio.async_mode)

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
    #print(idphoto)
    global dataPhoto
    #print(dataPhoto)
    #dataPhoto = jsonify(dataPhoto)
    #print(dataPhoto)
    print("============================================")
    return render_template('photo.html', id = idphoto)	

@app.route('/getdataPhoto', methods=['GET', 'POST'])
def getdataPhoto():
    global dataPhoto
    return str(dataPhoto)	
	
@app.route('/listphoto')
def listphoto():
    return render_template('listphoto.html')

@app.route('/biodata')
def biodata():
    return render_template('biodata.html')
	
@app.route('/listpost')
def post():
    return render_template('listpost.html')

@app.route('/weather')
def weather():
    return render_template('weathers.html')
		
@app.route('/getPhoto', methods=['GET', 'POST'])
def getPhoto():
    photo = graph.request("me?fields=albums{photos{images},name,created_time}")
    photo = jsonify(photo)
    return photo
@app.route('/getPhotobyAlbumsId', methods=['GET', 'POST'])
def getPhotobyAlbumsId():
    albumsid = request.get_data()
    albumsid = str(albumsid).replace("b", "")
    albumsid = str(albumsid).replace("'", "")
    #return albumsid
    #albumsid = request.json['albums']
    strrequest = "v3.0/"+albumsid+"?fields=photos{images}"
    #return strrequest
    photo = graph.request(strrequest)
    photo = jsonify(photo)
    return photo


@app.route('/getBiodata', methods=['GET', 'POST'])
def getBiodata():
    #id = graph.request("/me?fields=id")
    #name = graph.request("/me?fields=name")
    #photo = graph.request("/me?fields=photos{images}")
    bio = graph.request("me?fields=about,birthday,address,age_range,education,gender,hometown,name,relationship_status,email,picture")
    #print(photo)
    bio = jsonify(bio)
    return bio

@app.route('/getPost', methods=['GET', 'POST'])
def getPost():
    post = graph.request("me?fields=feed")
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

dataPhoto=""

@app.route('/updateDtaPhoto', methods=['POST'])
def updateDtaPhoto():
    global dataPhoto
    dataPhoto = request.json['albums']
    return str(dataPhoto)
''' Command '''
########              Call              ############

@app.route('/phone')
def phone():
    return render_template('phone.html')

@app.route('/listPhone', methods=['POST'])
def listPhone():
    with open("phoneLists.txt") as f:
        content = f.readlines()
    phoneData=[]   
    #userId =  request.form['userId']
    for c in range(len(content)):
        data = content[c].split(",")
        print(data)
        phoneData.append(content[c]) 
        '''
        phoneData[0] =  data[0]
        phoneData[1] =  data[1]
        phoneData[2] =  data[2]
        phoneData[3] =  data[3]
        '''
    return jsonify(phoneData)

#######################################################
########              Voice              ############
#######################################################
@app.route('/voice', methods=['GET', 'POST'])
def voice():
    global handin
    
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
        handin=res
        background_thread2()
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
#handin = False
handx=0
handy=0

@app.route('/cekHand', methods=['GET', 'POST'])	
def cekHand():
    global handin 
    global handx
    global handy 
    return jsonify({"0":handin,"1":handx,"2":handy})
    #return (str(handin))

initialState = 1
currentState = initialState
nextState = -1


def loadFSM(filename):
    fsmTable = []

    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            fsmTable.append(row)

    npFsmTable = np.asarray(fsmTable, dtype=str)
    # print(npFsmTable)

    return npFsmTable


def getGesture(stateName, fsmTable):
    gestures = []
    # print("========================================")
    # print("given state: ",stateName)
    for i in range(0, len(fsmTable)):
        if str(stateName) == str(fsmTable[i, 0]):
            gestures.append(fsmTable[i, 1])

    return np.array(gestures)


def getNextState(stateName, actionName, fsmTable):
    nextState = stateName
    print("========================================")
    print("given state: ", stateName)
    print("given action: ", actionName)
    for i in range(0, len(fsmTable)):
        if str(stateName) == str(fsmTable[i, 0]) and str(actionName) == str(fsmTable[i, 1]):
            # print("next state: ", fsmTable[i,2])
            nextState = fsmTable[i, 2]

    return nextState


def checkFSM(gesture, fsmTable):
    global currentState
    gesture = "-1"
    maxProb = -1
    idGest = -1

    currentState = getNextState(currentState, gesture, fsmTable)

    # print(currentState)

    return gesture


def checkhandIn(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image,is_in=False):
    global handx
    global handy 
    global handin
    handin = False
#    if is_in==False:
#        handin = False
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

        for i in range(boxXup, boxXdn):
            for j in range(boxYup, boxYdn):
                if depth_imageS.item(j, i) > dRefMaxF or depth_imageS.item(j, i) == 0:
                    boxColor.itemset((j, i, 0), 0)
                    boxColor.itemset((j, i, 1), 0)
                    boxColor.itemset((j, i, 2), 0)

        roi_boxCounter = boxColor[boxYup.__int__():boxYdn.__int__(), boxXup.__int__():boxXdn.__int__()]

        graybox = cv2.cvtColor(roi_boxCounter, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(graybox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        idx = 0
        for cnt in contours:
            #print(idx)
            idx += 1
            xb, yb, wb, hb = cv2.boundingRect(cnt)
            areaCnt = wb * hb
            #print(areaCnt)
            if areaCnt > 500:
                # print(areaCnt)
                handin = True
                handx = xb
                handy = yb
                    
                break
            idx += 1
    '''''
        if handin==True:
            cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                      (0, 255, 255), 2)
        else:
            cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                          (0, 0, 255), 2)
    '''''
    return handin
def getFace(faces):
    print(faces)

def translateAvailableGest(availableGest, classGest):

    ctr = 0
    translateGest = []
    translateIGest = []


    for i in range(0,len(classGest)):
        for j in range(0, len(availableGest)):
            if str(availableGest[j]) == classGest[i]:
                translateGest.append(i)
                ctr = ctr+1

    for i in range(0, 10):
        ctr2 = 0
        for j in range(0,len(translateGest)):
            if str(translateGest[j]) == str(i):
                ctr2 = 1
        if ctr2 == 0:
            translateIGest.append(i)
            #print(i)

    return translateGest, translateIGest

def manipWeight(weightI, gAvailIgnore, mulitp):

    maxClass = 10
    # change one column to 0 and see the probability result
    for b in range(maxClass):
        for a in range(len(gAvailIgnore)):
            if gAvailIgnore[a] == b:
                # weightDense2[:, b] = np.multiply(weightDense2[:, b],100)
                weightI[b] = np.multiply(weightI[b], mulitp)
    return weightI



@app.route('/gesture', methods=['GET', 'POST'])
def gesture(st="y"):
    detection_graph, sess = detector_utils.load_inference_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.2, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=320, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=180, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()
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
    global txt
    global txtLoad
    global txtDelay
    global txtRecord
    global txtDel
    global txtProbability
    global font
    global to_reload
    global app
    global currentState
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
    img_rows, img_cols, maxFrames = 50, 50, 55
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
    #classGest = ['1','11','12','13','4','5','7','8']
    classGest = ['1', '12', '13', '14', '15', '3', '4', '5', '8', '9']
    nameGest = ['call', 'scroll up', 'scroll down', 'right', 'left', "like", 'play/pause', 'close', 'click', 'back']
    delayGest = 10
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
    json_file = open('3dcnnresult/34/3dcnnmodel.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
	# load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    #loaded_model.load_weights("3dcnnresult/3dcnnmodel.hd5")
    loaded_model.load_weights("3dcnnresult/34/3dcnnmodel.hd5")
    loaded_model.compile(loss=categorical_crossentropy,
                     optimizer='adam', metrics=['accuracy'])
#setup cv face detection
    conf = loaded_model.model.get_config()


    shapeInput, ap = loaded_model.model.get_layer(name="dense_2").input_shape
    shapeOutput, sp = loaded_model.model.get_layer(name="dense_2").output_shape
    weightDense2 = loaded_model.model.get_layer(name="dense_2").get_weights()[0]
    weightDense22I = loaded_model.model.get_layer(name="dense_2").get_weights()[1]
    weightDense22A = loaded_model.model.get_layer(name="dense_2").get_weights()[1]
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    if(st=="x"):
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        '''
        profiles = config.resolve(pipeline)  
        device = profiles.get_device()
        ctx = rs.context()
        device.hardware_reset()
        gesture() 
        '''
        ytfsmfile = "youtube_fsm.txt"
        ytFsmTable = loadFSM(ytfsmfile)
    else:
        ytfsmfile = "fb_fsm.txt"
        ytFsmTable = loadFSM(ytfsmfile)
    updatedWeight = False
    updatedWeight2 = False
    config = rs.config()
    config.enable_stream(rs.stream.depth, cameraWidthR, cameraHeightR, rs.format.z16, frameRateR)
    config.enable_stream(rs.stream.color, cameraWidthR, cameraHeightR, rs.format.bgr8, frameRateR)
    '''vc = cv2.VideoCapture(0)
    rval , firstFrame = vc.read()
    heightc, widthc, depthcol = firstFrame.shape
    imgTxt = np.zeros((heightc, 400, 3), np.uint8)
    #print('tryyyyy1')'''
    stat =0

        
#        some_queue.put("something")
           
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
    num_frames = 0
    im_width, im_height = (848, 480)
    # max number of hands we want to detect/track
    num_hands_detect = 1
    score_thresh = 0.37
    while True:
        if True:
            dataImg = []

            # Wait for a coherent pair of frames: depth and color
            
            try:
                frames = pipeline.wait_for_frames()
            
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
    
                #depth_frame = frames.get_depth_frame()
                #color_frame = frames.get_color_frame()
    
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()
                
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
            
            except:
                pipeline.stop()
                print("Error wait_for_frames wait_for_frames")
                gesture()
            
            num_frames += 1
            image_np = color_image
            try:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            boxes, scores = detector_utils.detect_objects(
                    image_np, detection_graph, sess)
            for i in range(num_hands_detect):
                if (scores[i] > score_thresh):
                    handin = True
                    
                    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    break
            if handin == True:
                gestStart = True
            else:
                gestStart = False 
            # if not depth_frame or not color_frame:
            #if not color_frame:
                #continue

            # Convert images to numpy arrays
            #depth_image = np.asanyarray(depth_frame.get_data())
            #color_image = np.asanyarray(color_frame.get_data())

            if(backgroundRemove == True):
                # Remove background - Set pixels further than clipping_distance to grey
                #grey_color = 153
                grey_color = 0
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
                #draw_image = bg_removed
                color_image = bg_removed
                draw_image = color_image
            else:
                draw_image = color_image

            #face detection here
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#            faces = face_cascade.detectMultiScale(gray, 1.1, 2)
#
#            if len(faces) > 0 and gestCatch == False:
#                gestStart = True
#                x, y, w, h = faces[0]
#            else:
#                x, y, w, h = x, y, w, h
#
#            fArea = w*h
            
            if gestCatch == False:
                faces = face_cascade.detectMultiScale(gray, 1.1, 2)
                #print("face: ",len(faces))

                ctr = 0
                idxFace = -1
                minDist = 9999

                if len(faces) > 0:
                    for f in faces:
                        xh, yh, wh, hh = f
                        farea = wh * hh

                        midxf = int(xh + (wh * 0.5))
                        midyf = int(yh + (hh * 0.5))

                        depth_imageS = depth_image * depth_scale
                        deptRef = depth_imageS.item(midyf, midxf)

                        if deptRef <= minDist:
                            idxFace = ctr
                            minDist = deptRef

                        ctr = ctr+1
                    #print("id face", idxFace)

                    if idxFace >= 0:
                        x, y, w, h = faces[idxFace]
                        #cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #print(fArea)

            #if fArea > 3000:

            # crop the face then pass to resize
            #cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            midx = int(x + (w * 0.5))
            midy = int(y + (h * 0.5))

            xUp = (x - (w * 3))
            yUp = (y - (h * 1.5))

            xDn = ((x + w) + (w * 1))
            yDn = ((y + h) + (h * 2))

            if xUp < 1: xUp = 0
            if xDn >= 848: xDn = 848

            if yUp < 1: yUp = 0
            if yDn >= 480: yDn = 480

            #cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 0, 255), 2)

            #cv2.circle(draw_image, (midx.__int__(), midy.__int__()), 10, (255, 0, 0))
#            roi_color = color_image[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]
            roi_gray = gray[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]
            #find the depth of middle point of face
#            if handin == True and depthFrame==10:
#                depth_imageS = depth_image * depth_scale
#                deptRef = depth_imageS.item(midy, midx)
#                boxColor = color_image.copy()
#                checkhandIn(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image,handin)
            if backgroundRemove == True and gestCatch == False:

                depth_imageS = depth_image*depth_scale
                deptRef = depth_imageS.item(midy, midx)
                #print(clipping_distance)

                clipping_distance = (deptRef + 0.2)/depth_scale

                boxColor = color_image.copy()

#                handin = checkhandIn(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image)

#                if handin == True:
#                    gestStart = True
#                else:
#                    gestStart = False
                #print("handinnnnnnnnnnnn "+str(gestStart))
            if delayBol == False and gestStart == True:

                if depthFrame < maxFrames:
#                    frame = cv2.resize(roi_color, (img_rows, img_cols))
                    frame = cv2.resize(roi_gray, (img_rows, img_cols))
                    framearray.append(frame)
                    depthFrame = depthFrame + 1
                    txtLoad = txtLoad + "["
                    count=count+1
                    gestCatch = True

                    #print(depthFrame)


                if depthFrame == maxFrames:
                    dataImg.append(framearray)
                    xx = np.array(dataImg).transpose((0, 2, 3, 1))
                    X = xx.reshape((xx.shape[0], img_rows, img_cols, maxFrames, channel))
                    X = X.astype('float32')
                    #print('X_shape:{}'.format(X.shape))
                    #==================== Update the Weight =======================================
                    newWeightI = []
                    newWeightA = []
                    availableGest = getGesture(currentState, ytFsmTable)
                    availG, ignoreG = translateAvailableGest(availableGest, classGest)
                    if updatedWeight:
                        weightI = manipWeight(weightDense22I, ignoreG, 1000)

                        newWeightI.append(weightDense2)
                        newWeightI.append(weightI)

                    if updatedWeight2:
                        maxClass = 10
                        weightA = manipWeight(weightDense22A, availG, 1000)

                        newWeightA.append(weightDense2)
                        newWeightA.append(weightA)

                    #=================================================================================

                    if updatedWeight == False and updatedWeight2 == False:
                        newWeightI.append(weightDense2)
                        newWeightI.append(weightDense22A)

                    loaded_model.model.get_layer(name="dense_2").set_weights(newWeightI)
                    if handin==True:
                        # do prediction
                        resc = loaded_model.predict_classes(X)[0]
                        res = loaded_model.predict_proba(X)[0]
                        
                        # find the result of prediction gesture
                        resultC = classGest[resc]
                        nameResultG = nameGest[resc]

                        for a in range(0, len(res)):
                            print("Gesture {}: {} ".format(str(nameGest[a]), str(res[a] * 100)))
                    else:
                        resultC = 0
                        nameResultG = "not enough frame recorded"
                    print("===============================================================")

                    if updatedWeight2:
                        loaded_model.model.get_layer(name="dense_2").set_weights(newWeightA)

                        # do prediction
                        resc2 = loaded_model.predict_classes(X)[0]
                        res2 = loaded_model.predict_proba(X)[0]

                        # find the result of prediction gesture
                        resultC2 = classGest[resc2]
                        nameResultG2 = nameGest[resc2]
                    #imgTxt = np.zeros((480, 400, 3), np.uint8)                    
                    if updatedWeight2:
                        if res2[resc2] > res[resc]:
                            txt = "ignored gesture"
                            act = -1
                        else:
                            txt = nameResultG
                            act = resultC
                    else:
                        txt = nameResultG
                        act = resultC
                    #show text

                    # check with FSM for finding the next state
                    currentState = getNextState(currentState, act, ytFsmTable)
                    txt = "Gesture-" + str(resultC)
#                    txtProbability = str(res[resc]*100)+"%"

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
                sess.close()
                return act
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
#########                 Phone            ############
from twilio.rest import Client

@app.route('/getCall', methods=['GET', 'POST'])
def getCall():
    phoneNumber = ""
    phoneDial = ""
    SID = ""
    token = ""
    with open("phoneLists.txt") as f:
        content = f.readlines()
    print("========================") 
    print(request.form)    
    print("-----------------------")
    #userId =  request.json['userId']
    userId =  1
    data = content[userId].split(",")
     
    phoneNumber =  data[0]
    phoneDial =  data[1]
    SID =  data[2]
    token =  data[3]
    # Twilio phone number goes here. Grab one at https://twilio.com/try-twilio
    # and use the E.164 format, for example: "+12025551234"
    TWILIO_PHONE_NUMBER = str(phoneNumber)
    
    # list of one or more phone numbers to dial, in "+19732644210" format
    DIAL_NUMBERS = [str(phoneDial)]
    
    # URL location of TwiML instructions for how to handle the phone call
    TWIML_INSTRUCTIONS_URL = \
      "http://static.fullstackpython.com/phone-calls-python.xml"
    
    # replace the placeholder values with your Account SID and Auth Token
    # found on the Twilio Console: https://www.twilio.com/console
    client = Client(str(SID), str(token))
    dial_numbers(DIAL_NUMBERS,client,TWILIO_PHONE_NUMBER,DIAL_NUMBERS,TWIML_INSTRUCTIONS_URL)  

    return "call"

@app.route('/call/', methods=['GET', 'POST'])
def call():
    return render_template("call.html")

def dial_numbers(numbers_list,client,TWILIO_PHONE_NUMBER,DIAL_NUMBERS,TWIML_INSTRUCTIONS_URL):
    """Dials one or more phone numbers from a Twilio phone number."""
    for number in numbers_list:
        print("Dialing " + number)
        # set the method to "GET" from default POST because Amazon S3 only
        # serves GET requests on files. Typically POST would be used for apps
        client.calls.create(to=number, from_=TWILIO_PHONE_NUMBER,
                            url=TWIML_INSTRUCTIONS_URL, method="GET")
    return "dial"

@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
#    with thread_lock:
#        if thread is None:
#            thread = socketio.start_background_task(target=background_thread)
#    emit('my_response', {'data': 'Connected', 'count': 0})

def background_thread2():
    """Example of how to send server generated events to clients."""
    count = 0
    socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': handin},
                      namespace='/test')
 
def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': handin},
                      namespace='/test')
#######################################################       

if __name__ == '__main__':
    #gesture()
    #vc = cv2.VideoCapture(0)
    #app.run(debug = True)
    socketio.run(app, debug=True)
