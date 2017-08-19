#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
import copy
import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, defer
from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log

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
from sklearn.cluster import KMeans

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface
import pickle, pprint
import json
from numpy import genfromtxt
import requests
import pandas as pd


modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
trainDir=os.path.join(fileDir,'..','..','Train_Image')
classifierDir=os.path.join(fileDir,'..','..')
imagesDir=os.path.join(fileDir,'..','..')
peopleDir=os.path.join(fileDir,'..','..')
featureDir=os.path.join(fileDir,'..','..','Feature_gui')
Feature_unknown=os.path.join(fileDir,'..','..','Feature_unknown')
alignDir=os.path.join(fileDir,'..','..')
# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

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



def loadDist():
    try:
        with open(featureDir+'/classifier.pkl', 'rb') as f:
            # if sys.version_info[0] < 3:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
                return (le,clf)
            else:
                (le, clf) = pickle.load(f, encoding='latin1')
                return (le,clf)
    except Exception as e:
        return None

def loadUnknown():
    try:
        with open(Feature_unknown+'/classifier.pkl', 'rb') as f:
            # if sys.version_info[0] < 3:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
                return (le,clf)
            else:
                (le, clf) = pickle.load(f, encoding='latin1')
                return (le,clf)
    except Exception as e:
        return None




def retrain():
    os.chdir(alignDir)
    os.remove(alignDir+'/Aligned_data'+'/cache.t7')
    os.system('python align-dlib.py'+' '+trainDir+' '+'align outerEyesAndNose Aligned_data/ --size 96')
    os.system('./batch-represent/main.lua -outDir Feature_gui/ -data Aligned_data/')
    os.system('python classifier.py train Feature_gui/ --classifier RadialSvm')
    os.remove(alignDir+'/Aligned_data_unknown'+'/cache.t7')
    os.system('cp -r Feature_gui/classifier.pkl Feature_dir/')
    os.system('cp -r  Unknown/ Train_Image/')
    os.system('python align-dlib.py Train_Image/ align outerEyesAndNose Aligned_data_unknown/ --size 96')
    os.system('./batch-represent/main.lua -outDir Feature_unknown/ -data Aligned_data_unknown/')
    os.system('python classifier.py train Feature_unknown/ --classifier RadialSvm')
    os.remove(alignDir+'/Train_Image/'+'Unknown/')


    os.chdir(fileDir)

    try:
        with open(self.featureDir+'/classifier.pkl', 'rb') as f:
        # if sys.version_info[0] < 3:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
                return (le,clf)
            else:
                (le, clf) = pickle.load(f, encoding='latin1')
                return (le,clf)
    except Exception as e:
        return None





def loadImages():
    try:
        with open('images.pkl', 'rb') as f:
            # if sys.version_info[0] < 3:
            images = pickle.load(f)
        return images
    except Exception as e:
        return {}


#my_data = genfromtxt('people.csv', delimiter=',')
def loadModel():
    # model = open('model.pkl', 'r')
    # svm_persisted = pickle.load('model.pkl')
    # output.close()
    # return svm_persisted
    # return True
    try:
        with open('model.pkl', 'rb') as f:
            # if sys.version_info[0] < 3:
            mod = pickle.load(f)
            return mod
    except Exception as e:
        return None

def loadPeople():
    try:
        with open('people.pkl', 'rb') as f:
            mod = pickle.load(f)
            return mod
    except Exception as e:
        return []

class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        self.images = {}
        self.training = True
        self.people = []
        self.svm = None
        (self.le_dist,self.clf_dist)=loadDist()
        (self.le_unknown,self.clf_unknown)=loadUnknown()
        self.centroids_dist=pd.read_csv(featureDir+'/centroids_csv.csv')
        self.centroids_dist.drop(self.centroids_dist.columns[0],axis=1,inplace=True)
        self.embeddings_dist=pd.read_csv(featureDir+'/embeddings.csv')
        self.embeddings_unknown=pd.read_csv(Feature_unknown+'/embeddings.csv')
        self.centroids_unknown=pd.read_csv(Feature_unknown+'/centroids_csv.csv')
        self.centroids_unknown.drop(self.centroids_unknown.columns[0],axis=1,inplace=True)
        self.KMeans=None
        #self.centroids.drop(self.centroids.columns[0],axis=1,inplace=True)
        #self.retrain=False
        self.classifier="Distance"
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "ALL_STATE":
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            #if not self.training:
            #    self.trainSVM()
        elif msg['type'] == "ADD_PERSON":
            if msg['val'].encode('ascii','ignore') not in self.people:


               self.people.append(msg['val'].encode('ascii', 'ignore'))
               self.people=self.people
            #np.savetxt("people.csv", self.people, delimiter=",")
               with open(peopleDir+'/people.pkl', 'w') as f:
                    pickle.dump(self.people, f)

            print(self.people)
            if not os.path.isdir(trainDir+"/"+msg['val'].encode('ascii','ignore')):
                os.mkdir(trainDir+"/"+msg['val'].encode('ascii','ignore'))
            os.chdir(trainDir+"/"+msg['val'].encode('ascii','ignore'))

        elif msg['type'] == "UPDATE_IDENTITY":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == 'REQ_TSNE':
            self.sendTSNE(msg['people'])
        elif msg['type']== 'RE-TRAIN':
            print self.classifier
            self.retrain()
        elif msg['type']=='DISTANCE':
            #(self.le,self.clf)=loadDist()
            self.classifier="Distance"
            self.centroids=pd.read_csv(featureDir+'/centroids_csv.csv')
            self.centroids.drop(self.centroids.columns[0],axis=1,inplace=True)

            print self.classifier
        elif msg['type']=="UNKNOWN":
            #(self.le,self.clf)=loadUnknown()
            self.classifier="Unknown"
            self.centroids=pd.read_csv(Feature_unknown+'/centroids_csv.csv')
            self.centroids.drop(self.centroids.columns[0],axis=1,inplace=True)
            print self.classifier


        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

        if not training:
            self.trainSVM()

    def getData(self):
        X = []
        y = []
        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None

        if args.unknown:
            numUnknown = y.count(-1)
            numIdentified = len(y) - numUnknown
            numUnknownAdd = (numIdentified / numIdentities) - numUnknown
            if numUnknownAdd > 0:
                print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
                for rep in self.unknownImgs[:numUnknownAdd]:
                    # print(rep)
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def sendTSNE(self, people):
        #if self.classifier=='Distance':
        #    d=self.embeddings_dist
        #else:
        #    d=self.embeddings_unknown
        d = self.embeddings_dist
        if d is None:
            return
        else:
            print d.columns
            (X, y) = (copy.copy(d),copy.copy(d['labels']))
            X.drop('labels',axis=1, inplace=True)

        X_pca = PCA(n_components=50).fit_transform(X, X)
        tsne = TSNE(n_components=2, init='random', random_state=0)
        X_r = tsne.fit_transform(X_pca)

        yVals = list(np.unique(y))
        colors = cm.rainbow(np.linspace(0, 1, len(yVals)))

        # print(yVals)

        plt.figure()
        for c, i in zip(colors, yVals):
            #name = "Unknown" if i == -1 else people[i]
            name=i
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=name)
            plt.legend()

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)

        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "TSNE_DATA",
            "content": content
        }
        self.sendMessage(json.dumps(msg))

    def retrain(self):
        os.chdir(alignDir)
        
        
        os.system('python align-dlib.py'+' '+trainDir+' '+'align outerEyesAndNose Aligned_data/ --size 96')
        if os.path.isfile(alignDir+'/Aligned_data'+'/cache.t7'):

            os.remove(alignDir+'/Aligned_data'+'/cache.t7')
        os.system('./batch-represent/main.lua -outDir Feature_gui/ -data Aligned_data/')
        os.system('python classifier.py train Feature_gui/ --classifier RadialSvm')
        os.system('cp -r Feature_gui/classifier.pkl Feature_dir/')
        os.system('cp -r Feature_gui/centroids_csv.csv Feature_dir/')



        os.system('cp -r  Unknown/ Train_Image/')
        os.system('python align-dlib.py Train_Image/ align outerEyesAndNose Aligned_data_unknown/ --size 96')
        if os.path.isfile(alignDir+'/Aligned_data_unknown'+'/cache.t7'):

            os.remove(alignDir+'/Aligned_data_unknown'+'/cache.t7')
        os.system('./batch-represent/main.lua -outDir Feature_unknown/ -data Aligned_data_unknown/')
        os.system('python classifier.py train Feature_unknown/ --classifier RadialSvm')
        os.chdir(alignDir+'/Train_Image/')
        os.system('rm -rf Unknown/')
        


        os.chdir(fileDir)
        print self.classifier
        #if self.classifier=="Distance":


        try:
            with open(featureDir+'/classifier.pkl', 'rb') as f:
        # if sys.version_info[0] < 3:
                if sys.version_info[0] < 3:
                    (self.le_dist, self.clf_dist) = pickle.load(f)
                    #return (le,clf)
                else:
                    (self.le_dist, clf_dist) = pickle.load(f, encoding='latin1')
                        #return (le,clf)
        except Exception as e:
            return None

        #if self.classifier=="Unknown":
        try:
            with open(Feature_unknown+'/classifier.pkl', 'rb') as f:
        # if sys.version_info[0] < 3:
                if sys.version_info[0] < 3:
                    (self.le_unknown, self.clf_unknown) = pickle.load(f)
                        #return (le,clf)
                else:
                    (self.le_unknown, self.clf_unknown) = pickle.load(f, encoding='latin1')
                        #return (le,clf)
        except Exception as e:
            return None
        msg={"type":"RE-TRAINED"}
        self.sendMessage(json.dumps(msg))

        
        


        return







    def trainSVM(self):
        print("+ Training SVM on {} labeled images.".format(len(self.images)))
        d = self.getData()
        if d is None:
            self.svm = None
            return
        else:
            (X, y) = d
            numIdentities = len(set(y + [-1]))
            if numIdentities <= 1:
                return

            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            self.svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)
            print "Persisting Model", self.svm
            self.persistModel(self.svm)
            print "Loading Model"

    def loadModel(self):
        # model = open('model.pkl', 'r')
        # svm_persisted = pickle.load('model.pkl')
        # output.close()
        # return svm_persisted
        # return True
        with open(classifierDir+'/model.pkl', 'rb') as f:
            # if sys.version_info[0] < 3:
            mod = pickle.load(f)
            return mod

    def persistModel(self, mod):
        # output = open('model.pkl', 'w')
        with open(classifierDir+'/model.pkl', 'wb') as f:
            pickle.dump(mod, f)
        # svm_persisted = pickle.dump(mod, 'model.pkl', protocol=2)
        # output.close()
        return True



    def processFrame(self, dataURL, identity):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        if not self.training:
            annotatedFrame = np.copy(buf)

        # cv2.imshow('frame', rgbFrame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return


        identities = []

        if not self.training:

            bbs = align.getAllFaceBoundingBoxes(rgbFrame)
            
        else:
 
             bb = align.getLargestFaceBoundingBox(rgbFrame)
             bbs = [bb] if bb is not None else []
        content=[]
        for bb in bbs:
            # print(len(bbs))
            faces ={}
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue

            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
            if phash in self.images:
                identity = self.images[phash].identity
            else:
                rep = net.forward(alignedFace)
                # print(rep)
                if self.training:
                    self.images[phash] = Face(rep, identity)
                    # TODO: Transferring as a string is suboptimal.
                    # content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                    # fx=0.5, fy=0.5).flatten()]
                    content = [str(x) for x in alignedFace.flatten()]
                    msg = {
                        "type": "NEW_IMAGE",
                        "hash": phash,
                        "content": content,
                        "identity": identity,
                        "representation": rep.tolist()
                    }
                    self.sendMessage(json.dumps(msg))
                    #os.remove(alignDir+'/Aligned_data'+'/cache.t7')
                    personDir=os.getcwd()
                    with open (phash+'.jpg', 'wb') as handle:
                        #response = requests.get(image, stream=True)
                        #if not response.ok:
                        #    print response
                        #for block in response.iter_content(1024):
                        #    if not block:
                        #        break
                        #    handle.write(block)
                        handle.write(imgdata)
                    os.chdir(fileDir)
                    with open(imagesDir+'/images.pkl', 'w') as f:
                        pickle.dump(self.images, f)
                    os.chdir(personDir)


                else:
                    ####Prediction of Class using offline model with distance approach####
                    dist_distance={}
                    
                    for key in self.centroids_dist.keys():
                        dist_distance[key]=(np.linalg.norm(rep-list(self.centroids_dist[key])))
                    print self.centroids_dist.keys()

                    prediction_dist=self.clf_dist.predict_proba(rep).ravel()
                    #print prediction
                    maxI = np.argmax(prediction_dist)
                    person_dist = self.le_dist.inverse_transform(maxI)
                    confidence = prediction_dist[maxI]
                    distance=dist_distance[person_dist]
                    person_dist_min=min(dist_distance, key=dist_distance.get)
                    print person_dist

                    #print dist_distance
                    if dist_distance[person_dist]>0.70:
                        person_dist="unknown"
                    
                    ################################################################### 
                    dist_unknown={}
                    for key in self.centroids_unknown.keys():
                        dist_unknown[key]=(np.linalg.norm(rep-list(self.centroids_unknown[key])))
                    print self.centroids_unknown.keys()
                    prediction_unknown=self.clf_unknown.predict_proba(rep).ravel()
                    maxI = np.argmax(prediction_unknown)
                    person_unknown = self.le_unknown.inverse_transform(maxI)
                    confidence = prediction_unknown[maxI]
                    distance=dist_unknown[person_unknown]
                    #person_dist=min(dist, key=dist.get)
                    print dist_unknown
                    #if dist_unknown[person_unknown]>0.8:
                    #    person_unknown="unknown"
                    




#####################################################################################################################################
                    if len(self.people) == 0:
                        identity = -1
                    elif len(self.people) == 1:
                        identity = 0
                    elif self.svm:
                        identity = self.svm.predict(rep)[0]
                        print "predicted",identity

                    else:
                        print("hhh")
                        identity = -1
                    if identity not in identities:
                        identities.append(identity)
                        print identities

            if not self.training:
                bl = (bb.left(), bb.bottom())
                tr = (bb.right(), bb.top())
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                              thickness=3)
                for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
                    cv2.circle(annotatedFrame, center=landmarks[p], radius=3,
                               color=(102, 204, 255), thickness=-1)
                #if identity == -1:
                #    if len(self.people) == 1:
                #        name = self.people[0]

                #    else:
                #        name = "Unknown"
                #else:

                #    name = self.people[identity]
                if self.classifier=="Distance":
                    person_predicted=person_dist
                else:
                    person_predicted=person_unknown
                cv2.putText(annotatedFrame, person_predicted, (bb.left(), bb.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=(152, 255, 204), thickness=2)
                faces['identity']=person_predicted
                faces['left']=bb.left()
                faces['right']=bb.right()
                faces['top']=bb.top()
                faces['bottom']=bb.bottom()
                content.append(faces)



        if not self.training:
            msg = {
                "type": "IDENTITIES",
                "identities": identities
            }
            self.sendMessage(json.dumps(msg))

            plt.figure()
            plt.imshow(annotatedFrame)
            plt.xticks([])
            plt.yticks([])

            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, format='png')
            imgdata.seek(0)
            #content = 'data:image/png;base64,' + \
            #    urllib.quote(base64.b64encode(imgdata.buf))
            #msg = {
            #    "type": "ANNOTATED",
            #    "content": content
            #}
            #content={"identity":person_predicted,"bottom":bb.bottom(),"top":bb.top(),"left":bb.left(),"right":bb.right()}
            msg={
                "type":"ANNOTATED",
                "content":content
                }
            print "printing content",content
            self.sendMessage(json.dumps(msg))
            plt.close()

            #self.sendMessage(json.dumps(msg))


def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
