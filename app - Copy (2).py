from flask import Flask, jsonify, render_template, request
import traceback
from flask_restful import Resource, Api
import speech_recognition as sr
import webbrowser as wb
import time

import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle

from tensorflow.contrib.learn.python.learn.datasets import base
import pyrealsense2 as rs
import msvcrt
import time

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        #return {'hello': 'world'}
        #wb.open('http://127.0.0.1:5000/')
        return render_template('home.html')
        #return "http://127.0.0.1:5000/"

#api.add_resource(HelloWorld, '/')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/youtube/')
def youtube():
    return render_template('youtube.html')	


@app.route('/ajax1', methods=['GET', 'POST'])
def ajax1():
    try:
        user =  request.form.get('username')
        return "1"
    except Exception:
        return "error"
		
@app.route('/openapp', methods=['GET', 'POST'])	
def openapp():
    try:
        f = open ('result.txt','r')
        #print (f.read())
        return f.read()
    except Exception:
        return "error"

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

def drawLine(point1,point2,img):
    #print(np.asarray(point2))
    cv2.line(img, (point2[1],point2[0]), (point1[1],point1[0]), (255, 255, 255), thickness=30, lineType=8)
    return img
@app.route('/gesture', methods=['GET', 'POST'])
def gesture():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    xxx = np.zeros([480,640,3])
    count=0
    seq = np.zeros([480,640,3])
    points=[]

    graph = tf.Graph()



    train_path = 'letter_test/'
    test_path = 'letter_test/'

    # Training Parameters
    learning_rate = 0.0001 
    training_steps = 10000 
    batch_size = 4 
    display_step = 100
    num_hidden = 256

    # Network Parameters
    num_input = 28 # MNIST data input (img shape: 28*28)
    
    timesteps = 28 # timesteps
    #num_classes = len(classes)
    validation_size = .5
    test_label = '15'
    #datatesimages, datateslabel = dataset.read_test_set(test_path, img_size, classes, test_label)
    #logits = graph.get_tensor_by_name("logits:0")
    dirpath = 'letter_model'
    new_saver = tf.train.import_meta_graph(dirpath+'/my-model-10000.meta')
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    Y = tf.get_default_graph().get_tensor_by_name("Y:0")
    
    print(new_saver)
    logits = tf.get_collection("logits")[0]
    #logits = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
    print(logits)
    prediction = tf.nn.softmax(logits)
    
    with tf.Session() as sess:
        # Restore
    
        new_saver.restore(sess, dirpath+'/my-model-10000')
        
        #new_saver.restore(sess, 'my-save-dir/my-model-10000')
        blue = []
        try:
            
            while True:
                print("sssss")

            cv2.waitKey(1)
        finally:

            # Stop streaming
            pipeline.stop()
			
			

#######################################################
if __name__ == '__main__':
   app.run(debug = True)
