#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:04:03 2019

@author: hu
"""
import numpy as np
#import keras_applications as KA

class Config(object):
    def __init__(self, M):
        self.detectors = ['mrcnn', 'yolov3', 'dla_34', 'dla_34_new', 'dla_34_ext', 'yolov3-tiny', 'yolov4', 'yolov4-tiny', 'oim']
        self.backbones = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'Xception', 'MobileNet']
        assert M in self.detectors + self.backbones
        
        #configuration for the detectors
        self.M = M
        self.POOL_KERNEL = [24, 12]
        if M == 'mrcnn':
            self.BACKBONE = "resnet101"
            self.TOP_DOWN_PYRAMID_SIZE = 256
            self.TRAIN_BN = False
            self.POOL_SIZE = 7
            self.FC_SIZE = 520
            self.IMAGE_SIZE = [1024, 1024]
            self.IMAGE_RESIZE_MODE = "square"
            self.IMAGE_MIN_DIM = 800
            self.IMAGE_MAX_DIM = 1024
            self.IMAGE_MIN_SCALE = 0
            self.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
            self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
            self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
            self.GT_VR_THRESHOLD = 0.9
            self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + 81
            self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
            self.RPN_ANCHOR_STRIDE = 1
            self.RPN_NMS_THRESHOLD = 0.7
            self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
            self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
            self.POST_NMS_ROIS_INFERENCE = 1000
            self.DETECTION_MIN_CONFIDENCE = 0.90
            self.DETECTION_NMS_THRESHOLD = 0.5
            self.DETECTION_MAX_INSTANCES = 100
            self.PRE_NMS_LIMIT = 6000
            self.IMAGES_PER_GPU = 1
            self.layer = 1
            self.letter_box = True
            self.SEnet = True
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
        elif M == 'yolov3':
            self.MEAN_PIXEL = np.array([0, 0, 0])
            self.IMAGE_SIZE = [608, 608]
            self.RANDOM_SIZE = [416, 608, 800]
            self.IMAGE_MAX_DIM = max(self.IMAGE_SIZE)
            self.IMAGE_MIN_DIM = 384
            self.DETECTION_MIN_CONFIDENCE = 0.3
            self.DETECTION_NMS_THRESHOLD = 0.5
            self.YOLO_NUM_ANCHORS = 3
            self.YOLO_NUM_CLASSES = 1
            self.anchors = [[[12, 16], [19, 36], [40, 28]], [[36, 75], [76, 55], [72, 146]], [[142, 110], [192, 243], [459, 401]]]
            self.layer = 1
            self.letter_box = False
            self.SEnet = True
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
        elif M == 'dla_34':
            self.FILL_COLOR = (127.5, 127.5, 127.5)
            self.IMAGE_SIZE = [608, 1088]
            self.DETECTION_MIN_CONFIDENCE = 0.3
            self.heads = {'hm': 1, 'wh': 2, 'reg': 2}
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
            self.SEnet = True
            self.layer = 1
            self.ltrb = False
            self.MEAN_PIXEL = np.array([0, 0, 0])
        elif M == 'dla_34_ext':
            self.FILL_COLOR = (127.5, 127.5, 127.5)
            self.IMAGE_SIZE = [608, 1088]
            self.DETECTION_MIN_CONFIDENCE = 0.3
            self.heads = {'hm': 1, 'wh': 2, 'reg': 2}
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
            self.SEnet = True
            self.layer = 1
            self.ltrb = False
            self.MEAN_PIXEL = np.array([0, 0, 0])
        elif M == 'dla_34_new':
            self.FILL_COLOR = (127.5, 127.5, 127.5)
            self.IMAGE_SIZE = [608, 1088]
            self.DETECTION_MIN_CONFIDENCE = 0.3
            self.heads = {'hm': 1, 'wh': 4, 'reg': 2}
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
            self.SEnet = True
            self.layer = 1
            self.ltrb = True
            self.MEAN_PIXEL = np.array([0, 0, 0])
        elif M == 'yolov4' or M == 'yolov4-transfer':
            self.MEAN_PIXEL = np.array([0, 0, 0])
            self.IMAGE_SIZE = [608, 608]
            self.RANDOM_SIZE = [416, 608, 800]
            self.IMAGE_MAX_DIM = 608
            self.IMAGE_MIN_DIM = 578
            self.DETECTION_MIN_CONFIDENCE = 0.3
            self.DETECTION_NMS_THRESHOLD = 0.5
            self.YOLO_NUM_ANCHORS = 3
            self.YOLO_NUM_CLASSES = 1
            self.anchors = [[[12, 16], [19, 36], [40, 28]], [[36, 75], [76, 55], [72, 146]], [[142, 110], [192, 243], [459, 401]]]
            self.layer = 1
            self.letter_box = False
            self.SEnet = True
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
        elif M == 'yolov3-tiny':
            self.MEAN_PIXEL = np.array([0, 0, 0])
            self.IMAGE_SIZE = [416, 416]
            self.RANDOM_SIZE = [416, 608, 800]
            self.IMAGE_MAX_DIM = 416
            self.IMAGE_MIN_DIM = 384
            self.DETECTION_MIN_CONFIDENCE = 0.3
            self.DETECTION_NMS_THRESHOLD = 0.5
            self.YOLO_NUM_ANCHORS = 3
            self.YOLO_NUM_CLASSES = 1
            self.anchors = [[[10,14],  [23,27],  [37,58]],  [[81,82],  [135,169],  [344,319]]]
            self.layer = 1
            self.letter_box = False
            self.SEnet = True
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
        elif M == 'yolov4-tiny':
            self.MEAN_PIXEL = np.array([0, 0, 0])
            self.IMAGE_SIZE = [608, 608]
            self.RANDOM_SIZE = [416, 608, 800]
            self.IMAGE_MAX_DIM = 608
            self.IMAGE_MIN_DIM = 384
            self.DETECTION_MIN_CONFIDENCE = 0.3
            self.DETECTION_NMS_THRESHOLD = 0.5
            self.YOLO_NUM_ANCHORS = 3
            self.YOLO_NUM_CLASSES = 1
            self.anchors = [[[10,14],  [23,27],  [37,58]],  [[81,82],  [135,169],  [344,319]]]
            self.layer = 1
            self.letter_box = False
            self.SEnet = True
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
        elif M in self.backbones:
            '''
            self.MEAN_PIXEL = np.array([0, 0, 0])
            self.IMAGE_SIZE = [608, 608]
            self.RANDOM_SIZE = [416, 608, 800]
            self.layer = 1
            self.SEnet = True
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            self.l2_norm = self.sim_type != 'l2norm'
            self.net_dict = {'ResNet50':[KA.resnet.ResNet50, KA.resnet.preprocess_input],
              'ResNet101':[KA.resnet.ResNet101, KA.resnet.preprocess_input],
              'ResNet151':[KA.resnet.ResNet152, KA.resnet.preprocess_input],
              'ResNet50V2':[KA.resnet_v2.ResNet50V2, KA.resnet_v2.preprocess_input],
              'ResNet101V2':[KA.resnet_v2.ResNet101V2, KA.resnet_v2.preprocess_input],
              'ResNet151V2':[KA.resnet_v2.ResNet152V2, KA.resnet_v2.preprocess_input],
              'xception':[KA.xception.Xception, KA.xception.preprocess_input],
              'VGG16':[KA.vgg16.VGG16, KA.vgg16.preprocess_input],
              'VGG19':[KA.vgg19.VGG19, KA.vgg19.preprocess_input],
              'InceptionV3':[KA.inception_v3.InceptionV3, KA.inception_v3.preprocess_input],
              'InceptionResNetV2':[KA.inception_resnet_v2.InceptionResNetV2, KA.inception_resnet_v2.preprocess_input],
              'MobileNet':[KA.mobilenet.MobileNet, KA.mobilenet.preprocess_input],
              'MobileNetV2':[KA.mobilenet_v2.MobileNetV2, KA.mobilenet_v2.preprocess_input],
              'DenseNet121':[KA.densenet.DenseNet121, KA.densenet.preprocess_input],
              'DenseNet169':[KA.densenet.DenseNet169, KA.densenet.preprocess_input],
              'DenseNet201':[KA.densenet.DenseNet201, KA.densenet.preprocess_input],
              'NASNetMobile':[KA.nasnet.NASNetMobile, KA.nasnet.preprocess_input],
              'NASNetLarge': [KA.nasnet.NASNetLarge, KA.nasnet.preprocess_input],           
              }'''
            True
        elif M == 'oim': #unsupported
            self.MEAN_PIXEL = np.array([102.9801, 115.9465, 122.7717])
            self.IMAGE_SIZE = [600, 600]
            self.IMAGE_MAX_DIM = 600
            self.IMAGE_RESIZE_MODE = "square"
            self.IMAGE_MIN_SCALE = 0
            self.layer = 0
            self.sim_type = ['cosine', 'l2norm_cosine', 'l2norm', 'theta'][0]
            raise ValueError('unsupported type...')
        self.NUM_CLASSES = 81
        self.BATCH_SIZE = 6
        self.DEBUG = False
        self.MAX_INTERVAL = 80
        self.MAX_TRACKING_VOLUME = 35
        self.MIN_VR = 0.4
        self.MAX_OFFSET_LEN = 0.1
        self.IMAGE_RESIZE_MODE = "square"
        self.IMAGE_MIN_SCALE = 0
        self.DEBUG = False
        self.BATCH_SIZE = 6
        self.FC_SIZE = 1024
        self.TRAIN_BN = False
        self.FL = 256
        self.FN = 6
        self.DTABLE_SIZE = 5000
        self.max_volume = 8
        self.mgn = True
        self.mosaicSize = 4
        self.MAX_LOST_TRACK = 30
        self.LSTM_TIME_STEPS = 6

