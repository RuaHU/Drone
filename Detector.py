from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
'''uncomment to set the message level'''
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
'''uncomment to set the usage gpu'''
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
from tensorflow import Graph, Session

'''uncomment to set gpu usage'''
#from keras.backend.tensorflow_backend import set_session
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.12)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
#set_session(sess)

import cv2
import keras.layers as KL
import keras.backend as K
import numpy as np

from nets.YOLO import YOLO, CropAndResize
from nets.dla_34 import DLASeg

from tools.load_weights import load_weights_by_name
from tools.config import Config as BaseConfig
from tools.DataAugmentation import DataAugmentation

#alternate re-identification network
import torch
from nets.osnet_ain import osnet_drone_utils
from nets.keras_osnet import keras_osnet

from Package import Package
from typing import List
'''variable
'''



class Config(BaseConfig):
    '''
    some basic configuration information of the detection network.
    change the parameters to adopt custom detection senarios.
    '''
    def __init__(self, 
                 M : str):
        '''M: name of detection 
        M in ['yolov3', 'yolov4', 'yolov3-tiny', 'yolov4-tiny', 'dla_34', 'mrcnn']
        '''
        super(Config, self).__init__(M)
        
        '''configuration of the re-id network'''
        self.os_mean = [0.485, 0.456, 0.406]
        self.os_std = [0.229, 0.224, 0.225]
        
        #use the person search network to get the re-id features
        self.b_use_person_search_network = True
        #use addtional re-identification network to get re-id features
        #keras version
        self.b_use_keras_osnet = False
        #original pytorch version
        self.b_use_torch_osnet = False
        #do not use both of them
        assert not (self.b_use_keras_osnet and self.b_use_torch_osnet)
        
        self.M = M
        
        '''configuration of the detection network'''
        if self.M == 'yolov4-tiny':
            #the input image size
            self.IMAGE_MAX_DIM = 416
            self.IMAGE_SIZE = [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM]
            #the detection threshold, increase it to derease false detection
            self.DETECTION_MIN_CONFIDENCE = 0.2
            #the threshold for non-max-supression (NMS), increase the value to
            #decrease duplicated bounding boxes
            self.DETECTION_NMS_THRESHOLD = 0.3
            #whether to use multiple branch structure to get re-id feature
            #if mgn is False, the dimension of re-id feature will be 256,
            #otherwise will be 1536
            self.mgn = False
            
        if self.M == 'yolov4':
            self.DETECTION_MIN_CONFIDENCE = 0.2
            self.DETECTION_NMS_THRESHOLD = 0.3
            self.mgn = False
        
        if self.M == 'yolov3':
            self.DETECTION_MIN_CONFIDENCE = 0.2
            self.DETECTION_NMS_THRESHOLD = 0.3
            self.mgn = False
        
        if self.M == 'dla34':
            self.mgn = False
        

class Detector:
    '''
    this class is used to configure the detection network.
    simply passing parameter M to indicates the selecting detection network.
    '''
    def __init__(self, M : str):
        
        self.cfg = Config(M)
        
        self.da = DataAugmentation('validation', self.cfg)
        
        K.clear_session()
        
        thread_graph = Graph()
        
        #multiple threadinsg
        with thread_graph.as_default():
            self.thread_session = Session()
            with self.thread_session.as_default():
                
                if self.cfg.M in ['yolov3', 'yolov4', 'yolov3-tiny', 'yolov4-tiny']:
                    self.detector = YOLO(self.cfg)().model_drone()
                    load_weights_by_name(self.detector, 
                                         os.path.join('weights', self.cfg.M + '_reid.h5'))
                    
                if self.cfg.M == 'dla_34':
                    self.detector = DLASeg(self.cfg).model_drone()
                    load_weights_by_name(self.detector, 
                                         os.path.join('weights/dla_34_reid.h5'))
                self.graph = tf.get_default_graph()
        '''more detectors'''
        #if self.cfg.M == 'mrcnn':
            #self.detector = MRCNN(self.cfg).model_drone()
            #load_weights_by_name(self.detector,
                                 #os.path.join('weights/mrcnn_reid.h5'))
    
        
    def det(self, image):
        '''
        input: image, [W, H, 3]
        output: 
            1- re-identification feature vector
            2- detection bounding boxes
            3- detection scores
        '''
        image, _, _, meta = self.da(image.copy(), [])
        with self.graph.as_default():
            with self.thread_session.as_default():
                detection = self.detector.predict(np.stack([image], axis = 0))
        detection[1][0] = self.da.unmold(detection[1][0], meta)
        return detection
    
    def detPkg(self, 
               pkg : Package):
        image, _, _, meta = self.da(pkg.getImage(), [])
        with self.graph.as_default():
            with self.thread_session.as_default():
                detection = self.detector.predict(np.stack([image], axis = 0))
        detection[1][0] = self.da.unmold(detection[1][0], meta)
        pkg.detection = {'reid':detection[0][0],
                         'bbox':detection[1][0],
                         'scores':detection[2][0]}
        pkg.setStates(_bDetected = True)
    
    def detPkgs(self,
                pkgs : List[Package]):
        for pkg in pkgs: self.detPkg(pkg)
        '''
        _data = list(zip(*[self.da(pkg.getImage(), []) for pkg in pkgs]))
        with self.graph.as_default():
            with self.thread_session.as_default():
                detection = self.detector.predict([np.stack(_data[0], axis = 0)])
                
        print('detection:', detection[1])
        for i, pkg in enumerate(pkgs):
            pkg.detection = {'reid':detection[0][i],
                             'bbox':self.da.unmold(detection[1][i], _data[-1][i]),
                             'scores':detection[2][i]}
            pkg.setStates(_bDetected = True)'''
            
    @staticmethod
    def draw(image, bboxes):
        for box in bboxes:
            print(box)
            x, y, w, h = np.array(box).astype(np.int32)
            image = cv2.rectangle(image, 
                                  (x, y), 
                                  (x + w, y + h), 
                                  (255, 255, 255),
                                  2)
        return image
            
       
import threading
class detectionAndTrackingPool(threading.Thread):
    '''
    detect images once an image come into the detection queue
    '''
    def __init__(self,
                 _detector : Detector,
                 ):
        threading.Thread.__init__(self)
        self.detector = Detector('yolov4-tiny')
        self._detection_time_elispe = -1
        
    
    def run(self,):
        while True:
            image = cv2.imread('demo/1.jpg')
            detection = self.detector.det(image)
            print(detection)
        
import time
if __name__ == '__main__':
    #from sys import argv
    #detector = Detector(argv[1])
    #image = cv2.imread(argv[2])
    #detection = detector.det(image)
    detector = Detector('yolov4-tiny')
    image = cv2.imread('demo/1.jpg')
    detection = detector.det(image)
    image = Detector.draw(image, detection[1][0])
    cv2.imshow('detection', image)
    cv2.waitKey(0)
    print('re-identification feature \n', detection[0].shape)
    print('detection bounding boxes \n', detection[1].shape)
    print('detection scores \n', detection[2])
    
    
        
            
            
    
        
        
