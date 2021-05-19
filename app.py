
from __future__ import division, print_function
# coding=utf-8
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

import sys
import os
import glob
import re
import numpy as np
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,redirect,url_for,request,render_template
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app =Flask(__name__)


model_path ='model_cnn.h5'

model = load_model(model_path)

#model.make_predict_function() 

#preprocessing the step
def model_predict(img_path,model):
    
    img = image.load_img(img_path,target_size = (224,224))
    
   
    img_pred = image.img_to_array(img)
    #img_pred = np.expand_dims(x,axis=0)
    img = np.expand_dims(img_pred,axis =0)
     
    img=preprocess_input(img)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    prediction = model.predict_classes(img)
    return prediction
  


	# Constants:


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def upload():
    classes = {'TRAIN': ['NORMAL Your health is good', 'Chances of Pneumonia take doctor advice'],
	           'VALIDATION': ['BACTERIA', 'NORMAL'],
	           'TEST': ['BACTERIA', 'NORMAL', 'VIRUS']}

    


    if request.method=="POST":
        ##get the file from the post
        f = request.files['file']
        #save the file uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        prediction = model_predict(file_path, model)
        predicted_class = classes['TRAIN'][prediction[0]]
        print('We think that is {}.'.format(predicted_class.lower()))
        return str(predicted_class).lower()
    #return None

        
if __name__=='__main__':
    app.run(debug=True)
