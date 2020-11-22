from flask import Flask, render_template, send_from_directory, flash
import sys
sys.path.append("..")
from extract_cnn_vgg16_keras import VGGNet
import numpy as np

import h5py
import os
import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time
import shutil

import flask
from flask import *
#from flask_session import Session
from flask.json import jsonify
#from flask.json import flash
# import cv2
from werkzeug.utils import secure_filename

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

output="../features.h5"
model=VGGNet()
h5f = h5py.File(output,'r') # dont forget to close after use
#features are dataset1
imgFeats = h5f['dataset_1'][:]

#print(imgFeats)

imgNames = h5f['dataset_2'][:]
#print(imgNames)




APP_ROOT = os.path.dirname(os.path.abspath(__file__))


app=Flask(__name__)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def rmallfiles(folder):
	for filename in os.listdir(folder):
	    file_path = os.path.join(folder, filename)
	    try:
	        if os.path.isfile(file_path) or os.path.islink(file_path):
	            os.unlink(file_path)
	        elif os.path.isdir(file_path):
	            shutil.rmtree(file_path)
	    except Exception as e:
	        print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route("/",methods=['GET','POST'])
def home():
	target = os.path.join(APP_ROOT, 'images/')
	if not os.path.isdir(target):
		os.mkdir(target)
	else:
		print("images result directory already exists!")

	if request.method == 'POST':
		print("request recieved")
		rmallfiles("images/")
		f = request.files['file']
		if not (f and allowed_file(f.filename)):
			flash("File not supported. Allowed image formats : png,PNG, jpg, JPG, bmp")
			return jsonify({"error": 1001, "msg": "Allowed image formats : png,PNG, jpg, JPG, bmp"})

		#basepath = os.path.dirname(__file__)
		#upload_path = os.path.join(basepath, "static\\images", secure_filename(f.filename))
		filename = f.filename
		destination = "/".join([target, filename])
		f.save(destination)
		#img_url=os.path.join(basepath, 'static\\images', 'test.jpg')
		start_time=time.time()
		im = get_preditction(destination)
		print("Prediction cost", time.time() - start_time)
		print(im)
		print(filename)
		print("-------------------------------------------------------------")
		return render_template('searchResult.html',images=im,upload_image=filename,length=len(im))

	return render_template("searchEngine.html")



def get_preditction(img_url):
    print("--------------------------------------------------")
    print("               searching starts")
    print("--------------------------------------------------")

    # read and show query image
    basepath = os.path.dirname(__file__)  
    queryDir = img_url
    queryImg = mpimg.imread(queryDir)

    # extract query image's feature, compute simlarity score and sort
    queryVec = model.extract_feat(queryDir)
    scores = np.dot(queryVec, imgFeats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    # number of top retrieved images to show
    maxres = 15
    print(enumerate(rank_ID[0:maxres]))
    imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
    print(imlist[0])
    print("top %d images in order are: " % maxres, imlist)
    f1=open("../databaseClasses.txt","r")
    lines=f1.readlines()
    f1.close()
    Dict={}
    for i in lines:
    	l1=i.split('_')
    	fn=l1[1].split(" ")
    	Dict[l1[0]] = fn[0]
    imlist2 = []
    for i in imlist:
    	key=i.decode('UTF-8')[:3]
    	fn=Dict[key]
    	path="../256_ObjectCategories/"+fn+"/"+i.decode('UTF-8')
    	imlist2.append(path)
    	shutil.copy(path,"./images")
    #print(imlist2)
    return imlist


@app.route('/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


@app.route('/description', methods=['GET', 'POST'])
def description():
    
	#description page of images when clicked
    # show the form, it wasn't submitted
    return render_template('description.html')


@app.route("/about")
def about():
	return "<h1>coolkarni about<h1>"

if __name__=="__main__":

    app.run(debug=True)
