import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface

import  pprint
import pickle
import json
from numpy import genfromtxt
import requests
import pandas as pd

from flask import Flask,request,jsonify
app = Flask(__name__)

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
homeDir=os.path.join(fileDir,'..','..')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)
def loadModel():
    # model = open('model.pkl', 'r')
    # svm_persisted = pickle.load('model.pkl')
    # output.close()
    # return svm_persisted
    # return True
    with open(homeDir+'/model.pkl', 'rb') as f:
        # if sys.version_info[0] < 3:
        mod = pickle.load(f)
        return mod

def loadOfflineModel():

     with open(homeDir+'/Feature_dir/classifier.pkl', 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
                return (le,clf)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')
                return (le,clf)


def loadPeople():
     with open(homeDir+'/people.pkl', 'rb') as f:
        # if sys.version_info[0] < 3:
        mod = pickle.load(f)
        return mod

#classifier=loadModel()
#(le,clf) = loadOfflineModel()
#people=loadPeople()
def loadCentroids():

    with open(homeDir+'/Feature_dir/centroids.pkl','rb') as f:
        if sys.version_info[0] < 3:
            centroids=pickle.load(f)

            return centroids
        else:
            centroids=pickle.load(f, encoding='latin1')
            return centroids

#centroids=pd.read_csv(homeDir+'/Feature_dir/centroids_csv.csv')


dataURL="data:image/jpeg;base64,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"

#with open("r.json",'rb') as f:
#    kk = json.dumps(logs)
#    f.write(kk)
#    print "Json"




def processFrame(dataURL):
    head = "data:image/jpeg;base64,"
    #print dataURL
    #print '!!!!!!!!!!!!!'
    #req = requests.get(url='http://static6.businessinsider.com/image/537d34c2ecad04df68b0a788/roger-federer-shows-you-what-its-like-to-play-tennis-from-his-perspective-using-google-glass.jpg').content
    # print req.status_code
    #img = base64.b64encode(req)
    assert(dataURL.startswith(head))
    imgdata = base64.b64decode(dataURL[len(head):])
    imgF = StringIO.StringIO()
    imgF.write(imgdata)
    imgF.seek(0)
    img = Image.open(imgF)

    

    buf = np.fliplr(np.asarray(img))
    print buf.shape
    #resized_img=cv2.resize(buf,(300,400))
    rgbFrame = np.zeros(buf.shape, dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    #if not self.training:
    annotatedFrame = np.copy(buf)

    # cv2.imshow('frame', rgbFrame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     return
    (le,clf) = loadOfflineModel()
    centroids=pd.read_csv(homeDir+'/Feature_dir/centroids_csv.csv')
    identities = [] 
    bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        
    #if not self.training:

   	
    #else:

     #    bb = align.getLargestFaceBoundingBox(rgbFrame)
      #   bbs = [bb] if bb is not None else []
    #faces={}
    #faces.keys=[identity,left,right,bottom,top]
    for bb in bbs:
        # print(len(bbs))
        faces={}
        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                  landmarks=landmarks,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            continue

        phash = str(imagehash.phash(Image.fromarray(alignedFace)))
        #if phash in self.images:
        #    identity = self.images[phash].identity
        #else:
        rep = net.forward(alignedFace)
            # print(rep)
           # if self.training:
            #    self.images[phash] = Face(rep, identity)
                # TODO: Transferring as a string is suboptimal.
                # content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                # fx=0.5, fy=0.5).flatten()]
             #   content = [str(x) for x in alignedFace.flatten()]
              #  msg = {
               #     "type": "NEW_IMAGE",
                #    "hash": phash,
                #    "content": content,
                #    "identity": identity,
                #    "representation": rep.tolist()
                #}
                #print "training",identity
                #self.sendMessage(json.dumps(msg))
                #print "training",self.images
               # with open('images.json', 'w') as fp:
                #     json.dump(self.images, fp)
                #with open('images.pkl', 'w') as f:
                #    pickle.dump(self.images, f)
            #else:
            #    if len(self.people) == 0:
            #        identity = -1
            #    elif len(self.people) == 1:
            #        identity = 0
             #   elif self.svm:
        ##########################################ONLINE MODEL####################################################
        #print classifier.predict
        #identity = classifier.predict(rep)[0]
        #print "predicted",identity
        #name=people[identity]
        #print name
##################################################OFFLINE MODEL###################################################
        dist={}
            #print centroids['Abhishek']
        for key in centroids.keys():
            dist[key]=(np.linalg.norm(rep-list(centroids[key])))

        identity=clf.predict_proba(rep).ravel()
        maxI = np.argmax(identity)
        name = le.inverse_transform(maxI)
        confidence = identity[maxI]
        distance=dist[name]
        if dist[name]>0.7:
            name="Unknown"

        
        bl = (bb.left(), bb.bottom())
        tr = (bb.right(), bb.top())
        faces['identity']=name
        faces['left']=bb.left()
        faces['right']=bb.right()
        faces['top']=bb.top()
        faces['bottom']=bb.bottom()
        faces['confidence']=confidence
        faces['distance']= distance
        print faces
        identities.append(faces)
    

    #with open("r.json") as f:
    #    kk = json.loads(f.read())
    
    return identities



            
@app.route("/deleteModel", methods=['GET'])
def deleteMode():
	# TODO: fix these paths in EC2
	os.remove('/home/wlpt836/webface/model.pkl')
	os.remove('/home/wlpt836/webface/people.pkl')
	os.remove('/home/wlpt836/webface/images.pkl')
    os.remove(homeDir+'/Feature_dir/classifier.pkl')
    os.remove(homeDir+'/Feature_dir/centroids.pkl')
    #os.remove('/home/wlpt836/webface/people.pkl')
	return jsonify({"success": True})

@app.route("/api",methods=['POST'])
def api():
    content = request.get_json()
    req=content["url"]
    print req
    response_image = requests.get(req)
    print response_image
    uri = ("data:" + 
       response_image.headers['Content-Type'] + ";" +
       "base64," + base64.b64encode(response_image.content))
    #print uri
    return jsonify(processFrame(uri))

app.run(host='0.0.0.0',port=8000)
#print processFrame(dataURL)