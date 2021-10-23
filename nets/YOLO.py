#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:44:01 2019

@author: hu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from functools import wraps
from functools import reduce
import keras.layers as KL
import keras.backend as K
import keras.models as KM

from .BaseNet import *
    
class YOLOv3:
    def __init__(self, config, mode = 'training'):
        self.cfg = config
        self.mode = mode
        
    def darknet_body(self, inputs, training = None):
        reg[0] = None
        x = DarknetConv2D_BN_Leaky(32, (3, 3), name = None, training = training)(inputs)
        x = self.resblock_body(x, 64, 1, name = None, training = training)
        C2 = x = self.resblock_body(x, 128, 2, name = None, training = training)
        C3 = x = self.resblock_body(x, 256, 8, name = None, training = training)
        C4 = x = self.resblock_body(x, 512, 8, name = None, training = training)
        C5 = x = self.resblock_body(x, 1024, 4, name = None, training = training)
        if self.mode == 'training':
            return KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([C2, C3, C4, C5])
        else:
            return [C2, C3, C4, C5]
    
    def reid(self, inputs, training = False):
        if self.cfg.M == 'yolov3':
            feature_maps = self.darknet_body(inputs, training = training)
            feature_maps = self.proposal_map(feature_maps, training = training)
        elif self.cfg.M == 'yolov3-tiny':
            feature_maps = self.tiny(inputs, training = training)
        return feature_maps[:-2], feature_maps[-2:]
    
    def model(self, model_type):
        '''
        model_type: detection or reid
            detection:pure detection network
            reid: detection + reid
        '''
        input_image = KL.Input(shape = [None, None, 3], name ='input_image')
        input_bbox = KL.Input(shape = [None, 4], name = 'input_bbox')
        
        detection_map, reid_map = self.reid(input_image)
        detection_score, detection, regression, regression_scores = KL.Lambda(lambda x : detector(x, input_bbox, self.cfg))(detection_map)
        
        if model_type == 'detection':
            return KM.Model([input_image, input_bbox], [detection, detection_score])
        
        bboxes = KL.Lambda(lambda x : tf.concat([x[0], x[1][..., :4], x[2]], axis = 1))([input_bbox, regression, detection])
        reid_map = ATLnet(reid_map, layer = self.cfg.layer, SEnet = self.cfg.SEnet)
        reid_pooled = feature_pooling(self.cfg, name = 'alignedROIPooling')([bboxes] + reid_map)
        reid_pooled = KL.Lambda(lambda x : tf.squeeze(x, axis = 0))(reid_pooled)
        reid_vector = sMGN(reid_pooled, _eval = True, return_all = self.cfg.mgn, return_mgn = True, l2_norm = self.cfg.l2_norm)
        reid_vector = KL.Lambda(lambda x : tf.expand_dims(x[:, 0, 0, :], axis = 0))(reid_vector)
        prediction_vectors, regressed_vectors, detection_vectors = KL.Lambda(lambda x : [x[:, :tf.shape(input_bbox)[1], :], \
                                                                                     x[:, tf.shape(input_bbox)[1]:(2*tf.shape(input_bbox)[1]), :], \
                                                                                     x[:, (2*tf.shape(input_bbox)[1]):, :]])(reid_vector)
    
        return KM.Model([input_image, input_bbox], [prediction_vectors, regressed_vectors, regression, regression_scores, detection_vectors, detection, detection_score])


    def model_drone(self,):
        input_image = KL.Input(shape = [None, None, 3], name ='input_image')
        detection_map, reid_map = self.reid(input_image)
        detection, detection_score = KL.Lambda(lambda x : det(x, self.cfg), 
                                               name = 'det')(detection_map)
        
        reid_map = ATLnet(reid_map, layer = self.cfg.layer, SEnet = self.cfg.SEnet)
        reid_pooled = feature_pooling(self.cfg, name = 'alignedROIPooling')([detection] + reid_map)
        reid_pooled = KL.Lambda(lambda x : tf.squeeze(x, axis = 0))(reid_pooled)
        reid_vector = sMGN(reid_pooled, _eval = True, return_all = self.cfg.mgn, return_mgn = True, l2_norm = self.cfg.l2_norm)
        reid_vector = KL.Lambda(lambda x : tf.expand_dims(x[:, 0, 0, :], axis = 0),
                                name = 'reid')(reid_vector)
        
        return KM.Model([input_image], [reid_vector, detection, detection_score])

    def resblock_body(self, x, num_filters, num_blocks, strides = (2, 2), name = None, training = None):
        '''A series of resblocks starting with a downsampling Convolution2D'''
        # Darknet uses left and top padding instead of 'same' mode
        if name is None:
            names = [None]
        else:
            names = [name + str(1)]
        x = KL.ZeroPadding2D(((1,0),(1,0)))(x)
        x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=strides, name = names[0], training = training)(x)
        for i in range(num_blocks):
            if name is None:
                names = [None, None]
            else:
                names = [name + str(i) + str(1), name + str(i) + str(2)]
            y = compose(
                    DarknetConv2D_BN_Leaky(num_filters//2, (1,1), name = names[0], training = training),
                    DarknetConv2D_BN_Leaky(num_filters, (3,3), name = names[1], training = training))(x)
            x = KL.Add()([x,y])
        return x
    
    def make_last_layers(self, x, num_filters, out_filters, name = None, training = None):
        '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
        if name is None:
            names = [None, None, None, None, None, None, None]
        else:
            names = [name + str(1), name + str(2), name + str(3), name + str(4), name + str(5), name + str(6), name + str(7)]
        x = compose(
                DarknetConv2D_BN_Leaky(num_filters, (1,1), name = names[0], training = training),
                DarknetConv2D_BN_Leaky(num_filters*2, (3,3), name = names[1], training = training),
                DarknetConv2D_BN_Leaky(num_filters, (1,1), name = names[2], training = training),
                DarknetConv2D_BN_Leaky(num_filters*2, (3,3), name = names[3], training = training),
                DarknetConv2D_BN_Leaky(num_filters, (1,1), name = names[4], training = training))(x)
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters*2, (3,3), name = names[5], training = training),
                DarknetConv2D(out_filters, (1,1), name = names[6]))(x)
        return x, y
    
    def proposal_map(self, inputs, training = None):
        '''pyramid feature maps yolov3'''
        reg[0] = None
        C2, C3, C4, C5 = inputs
        x, P5 = self.make_last_layers(C5, 512, self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), training = training)
        x = compose(
                DarknetConv2D_BN_Leaky(256, (1,1), training = training),
                KL.UpSampling2D(2))(x)
        x = KL.Concatenate()([x,C4])
        x, P4 = self.make_last_layers(x, 256, self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), training = training)
        x = compose(
                DarknetConv2D_BN_Leaky(128, (1,1), training = training),
                KL.UpSampling2D(2))(x)
        x = KL.Concatenate()([x,C3])
        x, P3 = self.make_last_layers(x, 128, self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), training = training)
        if self.model == 'training':
            [P3, P4, P5, x] = KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([P3, P4, P5, x])
        reg[0] = l1l2_reg
        #reid feature map
        x = compose(
                DarknetConv2D_BN_Leaky(64, (1,1), training = False),
                KL.UpSampling2D(2))(x)
        C2_enhance = KL.Concatenate(name = 'vis_concatenate')([x,C2])
        
        return [P3, P4, P5, C2, C2_enhance]
    
    def tiny(self, inputs, training = False):
        reg[0] = None
        c1 = x = compose(
                DarknetConv2D_BN_Leaky(16, (3,3), name = None, training = training),
                KL.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                DarknetConv2D_BN_Leaky(32, (3,3), name = None, training = training),
                KL.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                DarknetConv2D_BN_Leaky(64, (3,3), name = None, training = training))(inputs)
        
        c2 = x1 = compose(
                KL.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                DarknetConv2D_BN_Leaky(128, (3,3), name = None, training = training),
                KL.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                DarknetConv2D_BN_Leaky(256, (3,3), name = None, training = training))(x)
        
        x2 = compose(
                KL.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                DarknetConv2D_BN_Leaky(512, (3,3), name = None, training = training),
                KL.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
                DarknetConv2D_BN_Leaky(1024, (3,3), name = None, training = training),
                DarknetConv2D_BN_Leaky(256, (1,1), name = None, training = training))(x1)
        y1 = compose(
                DarknetConv2D_BN_Leaky(512, (3,3), name = None, training = training),
                DarknetConv2D(self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), (1,1), name = None))(x2)
    
        x = compose(
                DarknetConv2D_BN_Leaky(128, (1,1), name = None, training = training),
                KL.UpSampling2D(2))(x2)
        
        c2_enhance = KL.Concatenate()([x, c2])
        x = DarknetConv2D_BN_Leaky(256, (3,3), name = None, training = training)(c2_enhance)
        y2 = DarknetConv2D(self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), (1,1), name = None)(x)

        if self.mode == 'training':
            c2_enhance = KL.Lambda(lambda x : tf.stop_gradient(x))(c2_enhance)
            x = KL.Lambda(lambda x : tf.stop_gradient(x))(x)
            c1 = KL.Lambda(lambda x : tf.stop_gradient(x))(c1)
        
        x = compose(
                DarknetConv2D_BN_Leaky(128, (1,1), name = None, training = training),
                KL.UpSampling2D(2))(x)
        
        c1_enhance = KL.Concatenate()([x, c1])

        if self.mode == 'training':
            y1, y2, x1, x2 = KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([y1, y2, x1, x2])
        return [y2, y1, c2_enhance, c1_enhance]
    
    
class YOLOv4:
    def __init__(self, config):
        self.cfg = config
    
    def darknet_body(self, inputs, training = None):
        reg[0] = None
        x = DarknetConv2D_BN_Mish(32, (3,3), name = None, training = training)(inputs) #3
        x = self.resblock_body(x, 64, 1, all_narrow = False, name = None, training = training) # 1 + 3 * 3 + 1 * (2 * 3 + 1) + 3 + 1 + 3 = 24-->27
        C2 = x = self.resblock_body(x, 128, 2, name = None, training = training) # 1 + 3 * 3 + 2 * (2 * 3 + 1) + 3 + 1 + 3 = 31-->58
        C3 = x = self.resblock_body(x, 256, 8, name = None, training = training) # 1 + 3 * 3 + 8 * (2 * 3 + 1) + 3 + 1 + 3 = 17 + 56 = 73-->131
        C4 = x = self.resblock_body(x, 512, 8, name = None, training = training) # 1 + 3 * 3 + 8 * (2 * 3 + 1) + 3 + 1 + 3 = 17 + 56 = 73-->204
        C5 = x = self.resblock_body(x, 1024, 4, name = None, training = training) # 1 + 3 * 3 + 4 * (2 * 3 + 1) + 3 + 1 + 3 = 17 + 28 = 45-->249
        return KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([C2, C3, C4, C5])

    def reid(self, inputs, training = False):
        if self.cfg.M == 'yolov4':
            feature_maps = self.darknet_body(inputs, training = training)
            feature_maps = self.proposal_map(feature_maps, training = training)
        elif self.cfg.M == 'yolov4-tiny':
            feature_maps = self.tiny(inputs, training = training)
        return feature_maps[:-2], feature_maps[-2:]
    
    def model_drone(self,):
        input_image = KL.Input(shape = [None, None, 3], name ='input_image')
        detection_map, reid_map = self.reid(input_image)
        detection, detection_score = KL.Lambda(lambda x : det(x, self.cfg), 
                                               name = 'det')(detection_map)
        
        reid_map = ATLnet(reid_map, layer = self.cfg.layer, SEnet = self.cfg.SEnet)
        reid_pooled = feature_pooling(self.cfg, name = 'alignedROIPooling')([detection] + reid_map)
        reid_pooled = KL.Lambda(lambda x : tf.squeeze(x, axis = 0))(reid_pooled)
        reid_vector = sMGN(reid_pooled, _eval = True, return_all = self.cfg.mgn, return_mgn = True, l2_norm = self.cfg.l2_norm)
        reid_vector = KL.Lambda(lambda x : tf.expand_dims(x[:, 0, 0, :], axis = 0),
                                name = 'reid')(reid_vector)
        
        return KM.Model([input_image], [reid_vector, detection, detection_score])
    
    def model_drone_masked(self,):
        input_image = KL.Input(shape = [None, None, 3], name ='input_image')
        reid_masks = KL.INput(shape = [1], name = 'reid_masks')
                           
        detection_map, reid_map = self.reid(valid_images)
        detection, detection_score = KL.Lambda(lambda x : det_batch(x, self.cfg), 
                                               name = 'det')(detection_map)
        
        reid_map = ATLnet(reid_map, layer = self.cfg.layer, SEnet = self.cfg.SEnet)
        reid_pooled = feature_pooling(self.cfg, name = 'alignedROIPooling')([detection] + reid_map)
        reid_pooled = KL.Lambda(lambda x : tf.squeeze(x, axis = 0))(reid_pooled)
        reid_vector = sMGN(reid_pooled, _eval = True, return_all = self.cfg.mgn, return_mgn = True, l2_norm = self.cfg.l2_norm)
        reid_vector = KL.Lambda(lambda x : tf.expand_dims(x[:, 0, 0, :], axis = 0),
                                name = 'reid')(reid_vector)
        
        return KM.Model([input_image, reid_masks], [reid_vector, detection, detection_score])
        

    def model(self, model_type):
        input_image = KL.Input(shape = [None, None, 3], name ='input_image')
        input_bbox = KL.Input(shape = [None, 4], name = 'input_bbox')
        detection_map, reid_map = self.reid(input_image)
        detection_score, detection, regression, regression_scores = KL.Lambda(lambda x : detector(x, input_bbox, self.cfg))(detection_map)
        
        if model_type == 'detection':
            return KM.Model([input_image, input_bbox], [detection, detection_score])
        elif model_type == 'reid':
            bboxes = KL.Lambda(lambda x : tf.concat([x[0], x[1][..., :4], x[2]], axis = 1))([input_bbox, regression, detection])
            reid_map = ATLnet(reid_map, layer = self.cfg.layer, SEnet = self.cfg.SEnet)
            reid_pooled = feature_pooling(self.cfg, name = 'alignedROIPooling')([bboxes] + reid_map)
            reid_pooled = KL.Lambda(lambda x : tf.squeeze(x, axis = 0))(reid_pooled)
            reid_vector = sMGN(reid_pooled, _eval = True, return_all = self.cfg.mgn, return_mgn = True, l2_norm = self.cfg.l2_norm)
            reid_vector = KL.Lambda(lambda x : tf.expand_dims(x[:, 0, 0, :], axis = 0))(reid_vector)
            prediction_vectors, regressed_vectors, detection_vectors = KL.Lambda(lambda x : [x[:, :tf.shape(input_bbox)[1], :], \
                                                                                         x[:, tf.shape(input_bbox)[1]:(2*tf.shape(input_bbox)[1]), :], \
                                                                                         x[:, (2*tf.shape(input_bbox)[1]):, :]])(reid_vector)
        
            return KM.Model([input_image, input_bbox], [prediction_vectors, regressed_vectors, regression, regression_scores, detection_vectors, detection, detection_score])
        
    def proposal_map(self, inputs, training = None):
        '''pyramid feature maps yolov4'''
        reg[0] = None
        #152, 76, 38, 19
        [C2, C3, C4, C5] = inputs
        
        P5 = self.make_last_layers(C5, 512, 255, bspp = True, training = training)
    
        P5_up = compose(DarknetConv2D_BN_Leaky(256, (1,1), training = training), KL.UpSampling2D(2))(P5)
        
        P4 = DarknetConv2D_BN_Leaky(256, (1,1), training = training)(C4)
        P4 = KL.Concatenate()([P4, P5_up])
        P4 = self.make_last_layers(P4, 256, 255, training = training)
        #up 4
        P4_up = compose(DarknetConv2D_BN_Leaky(128, (1,1), training = training), KL.UpSampling2D(2))(P4)
        
        P3 = DarknetConv2D_BN_Leaky(128, (1,1), training = training)(C3)
        P3 = KL.Concatenate()([P3, P4_up])
        P3, P3_output = self.make_last_layers(P3, 128, self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), training = training, last = True)
    
        P3 = KL.Lambda(lambda x : tf.stop_gradient(x))(P3)
        #reid feature map
        reg[0] = l1l2_reg
        P3_up = compose(DarknetConv2D_BN_Leaky(64, (1,1), name = 'yfm_fn1', training = None), KL.UpSampling2D(2))(P3)
        C2_enhance = KL.Concatenate()([C2, P3_up])
        
        reg[0] = None
        P3_down = KL.ZeroPadding2D(((1,0),(1,0)))(P3)
        P3_down = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2), training = training)(P3_down)
        P4 = KL.Concatenate()([P3_down, P4])
        P4, P4_output = self.make_last_layers(P4, 256, self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), training = training, last = True)
    
        P4_down = KL.ZeroPadding2D(((1,0),(1,0)))(P4)
        P4_down = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2), training = training)(P4_down)
        P5 = KL.Concatenate()([P4_down, P5])
        P5, P5_output = self.make_last_layers(P5, 512, self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), training = training, last = True)
        [P3, P4, P5] = KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([P3_output, P4_output, P5_output])
        return [P3, P4, P5, C2, C2_enhance]

    def resblock_body(self, x, num_filters, num_blocks, strides = (2, 2), name = None, training = None, all_narrow=True):
        '''A series of resblocks starting with a downsampling Convolution2D'''
        # Darknet uses left and top padding instead of 'same' mode
        if name is None:
            gnames = [None for i in range(5)]
        else:
            gnames = [name + str(i) for i in range(5)]
        preconv1 = KL.ZeroPadding2D(((1,0),(1,0)))(x)
        preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=strides, name = gnames[0], training = training)(preconv1)
        shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), name = gnames[1], training = training)(preconv1)
        mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), name = gnames[2], training = training)(preconv1)
        for i in range(num_blocks):
            if name is None:
                lnames = [None, None]
            else:
                lnames = [name + str(i) + str(1), name + str(i) + str(2)]
            y = compose(
                    DarknetConv2D_BN_Mish(num_filters//2, (1,1), name = lnames[0], training = training),
                    DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3), name = lnames[1], training = training))(mainconv)
            mainconv = KL.Add()([mainconv,y])
        postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), name = gnames[3], training = training)(mainconv)
        route = KL.Concatenate()([postconv, shortconv])
        return DarknetConv2D_BN_Mish(num_filters, (1,1), name = gnames[4], training = training)(route)

    def make_last_layers(self, x, num_filters, out_filters, bspp = False, name = None, training = None, last = False):
        '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
        if name is None:
            names = [None for i in range(8)]
        else:
            names = [name + str(i) for i in range(8)]
        x = compose(
                DarknetConv2D_BN_Leaky(num_filters, (1,1), name = names[0], training = training),
                DarknetConv2D_BN_Leaky(num_filters*2, (3,3), name = names[1], training = training),
                DarknetConv2D_BN_Leaky(num_filters, (1,1), name = names[2], training = training),
                spp(spp = bspp, name = names[7], training = training),
                DarknetConv2D_BN_Leaky(num_filters*2, (3,3), name = names[3], training = training),
                DarknetConv2D_BN_Leaky(num_filters, (1,1), name = names[4], training = training))(x)
        if not last:return x
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters*2, (3,3), name = names[5], training = training),
                DarknetConv2D(out_filters, (1,1), name = names[6]))(x)
        return x, y
    
    def tiny_body(self, inputs, num_filters, training = False):
        x1 = x = DarknetConv2D_BN_Leaky(num_filters, (3,3), name = None, training = training)(inputs)
        x = KL.Lambda(lambda x : tf.split(x, 2, axis = -1)[1])(x)
        x2 = x = DarknetConv2D_BN_Leaky(num_filters//2, (3,3), name = None, training = training)(x)
        x = DarknetConv2D_BN_Leaky(num_filters//2, (3,3), name = None, training = training)(x)
        x = KL.Concatenate()([x, x2])
        y = x = DarknetConv2D_BN_Leaky(num_filters, (1,1), name = None, training = training)(x)
        x = KL.Concatenate()([x1, x])
        x = KL.MaxPooling2D(pool_size=(2,2), strides = (2, 2))(x)
        return x, y
    
    def tiny_3l(self, inputs, training = False):
        reg[0] = None
        x = KL.ZeroPadding2D(((1,0),(1,0)))(inputs)
        x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2), name = None, training = training)(x)
        x = KL.ZeroPadding2D(((1,0),(1,0)))(x)
        x = DarknetConv2D_BN_Leaky(64, (3,3), strides=(2,2), name = None, training = training)(x)
        
        x, c1 = self.tiny_body(x, 64, training = training)
        x, c2 = self.tiny_body(x, 128, training = training)
        x, c3 = self.tiny_body(x, 256, training = training)
        
        x2 = DarknetConv2D_BN_Leaky(512, (3,3), name = None, training = training)(x)
        
        x2 = x = DarknetConv2D_BN_Leaky(256, (1,1), name = None, training = training)(x2)
        x = DarknetConv2D_BN_Leaky(512, (3,3), name = None, training = training)(x)
        p1 = DarknetConv2D(self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), (1,1), name = None)(x)
        
        x = compose(DarknetConv2D_BN_Leaky(128, (1,1), name = None, training = training), KL.UpSampling2D(2))(x2)
        
        c3_enhance = KL.Concatenate()([x, c3])
        
        x3 = x = DarknetConv2D_BN_Leaky(256, (3,3), name = None, training = training)(c3_enhance)
        p2 = DarknetConv2D(self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), (1,1), name = None)(x)
        
        
        x = compose(DarknetConv2D_BN_Leaky(64, (1,1), name = None, training = training), KL.UpSampling2D(2))(x3)
        
        c2_enhance = KL.Concatenate()([x, c2])
        
        x4 = x =  DarknetConv2D_BN_Leaky(128, (3,3), name = None, training = training)(c2_enhance)
        p3 = DarknetConv2D(self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), (1,1), name = None)(x)
        
        c1, x = KL.Lambda(lambda x : [tf.stop_gradient(f) for f in x])([c1, x])
        x = compose(DarknetConv2D_BN_Leaky(32, (1,1), name = None, training = training), KL.UpSampling2D(2))(x4)
        c1_enhance = KL.Concatenate()([x, c1])
        
        p1, p2 = KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([p1, p2])
        
        return [p3, p2, p1, c2_enhance, c1_enhance]
    
    
    def tiny(self, inputs, training = False):
        reg[0] = None
        x = KL.ZeroPadding2D(((1,0),(1,0)))(inputs)
        x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2), name = None, training = training)(x)
        x = KL.ZeroPadding2D(((1,0),(1,0)))(x)
        x = DarknetConv2D_BN_Leaky(64, (3,3), strides=(2,2), name = None, training = training)(x)
        
        x, c1 = self.tiny_body(x, 64, training = training)
        x, c2 = self.tiny_body(x, 128, training = training)
        x, c3 = self.tiny_body(x, 256, training = training)
        
        x2 = DarknetConv2D_BN_Leaky(512, (3,3), name = None, training = training)(x)
        
        x2 = x = DarknetConv2D_BN_Leaky(256, (1,1), name = None, training = training)(x2)
        x = DarknetConv2D_BN_Leaky(512, (3,3), name = None, training = training)(x)
        p1 = DarknetConv2D(self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), (1,1), name = None)(x)
        
        x = compose(DarknetConv2D_BN_Leaky(128, (1,1), name = None, training = training), KL.UpSampling2D(2))(x2)
        
        c3_enhance = KL.Concatenate()([x, c3])
        
        x = DarknetConv2D_BN_Leaky(256, (3,3), name = None, training = training)(c3_enhance)
        p2 = DarknetConv2D(self.cfg.YOLO_NUM_ANCHORS*(self.cfg.YOLO_NUM_CLASSES+5), (1,1), name = None)(x)
        
        c1, c2, x = KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([c1, c2, x])
        reg[0] = l1l2_reg
        x = compose(DarknetConv2D_BN_Leaky(64, (1,1), name = None, training = training), KL.UpSampling2D(2))(x)
        
        c2_enhance = KL.Concatenate()([x, c2])
        
        x =  DarknetConv2D_BN_Leaky(128, (3,3), name = None, training = training)(c2_enhance)
        x = compose(DarknetConv2D_BN_Leaky(32, (1,1), name = None, training = training), KL.UpSampling2D(2))(x)
        c1_enhance = KL.Concatenate()([x, c1])
        
        p1, p2 = KL.Lambda(lambda x : [K.stop_gradient(f) for f in x])([p1, p2])
        
        return [p2, p1, c2_enhance, c1_enhance]

def detector(feature_maps, boxes, cfg):
    '''
    feature maps: [feature_map1, feature_map2, feature_map3]
        feature_mapi: [1, N, N, 255]
    boxes: [1, Nb, 4]
    anchors: [anchor1, anchor2, anchor3]
        anchori: [3, 2]
    tasks:
        1: get detection (dets, detscores)
        2: get confidence for the input boxes (bboxes, bboxes_score)
        3: search for regressed boxes and the confidence for input boxes (rbboxes, rbboxes_score)
    '''
    boxes = boxes[0, ...]
    dets, detscores = [], []
    bboxes, bboxes_score = [], []
    for i, feats in enumerate(feature_maps):
        anchors_tensor = K.reshape(K.constant(cfg.anchors[i]), [1, 1, 3, 2])
        features = feats[0, ...]
        shape = K.shape(features)[:2]
        grid = tf.cast(tf.expand_dims(tf.stack(tf.meshgrid(tf.range(0, shape[1]), tf.range(0, shape[0])), axis = -1), axis = 2), K.dtype(features))
        features = K.reshape(features, [shape[0], shape[1], cfg.YOLO_NUM_ANCHORS, cfg.YOLO_NUM_CLASSES + 5])
        det_xy = (K.sigmoid(features[..., :2]) + grid) / K.cast(shape[::-1], K.dtype(features))
        det_wh = K.exp(features[..., 2:4]) * anchors_tensor / K.cast(K.constant(cfg.IMAGE_SIZE), K.dtype(features))
        detyxyx = K.concatenate([det_xy[..., ::-1] - det_wh[..., ::-1]/2., det_xy[..., ::-1] + det_wh[..., ::-1]/2.], axis = -1)
        det_conf = K.sigmoid(features[..., 4:5])
        det_prob = K.sigmoid(features[..., 5:])
        det_scores = det_conf * det_prob
        
        cboxes = K.cast((boxes[:, :2] + boxes[:, 2:]) * K.reshape(K.cast(shape, dtype = 'float32'), [-1, 2]) / 2, dtype = 'int32')
        bboxes_score.append(tf.gather_nd(det_scores[:, :, :, 0], cboxes))
        
        bboxes.append(tf.gather_nd(detyxyx, cboxes))
        
        detyxyx = K.reshape(detyxyx, [-1, 4])
        det_scores = K.reshape(det_scores, [-1, cfg.YOLO_NUM_CLASSES])
        dets.append(detyxyx)
        detscores.append(det_scores)
    
    bboxes_score = K.concatenate(bboxes_score, axis = 1)
    bboxes = K.concatenate(bboxes, axis = 1)
    index = K.concatenate([K.reshape(K.arange(0, stop = K.shape(bboxes_score)[0]), [-1, 1]), \
                           K.reshape(tf.cast(tf.argmax(K.reshape(bboxes_score, [K.shape(bboxes_score)[0], -1]), \
                                               axis = -1), 'int32'), [-1, 1])], axis = -1)

    bboxes_score = K.reshape(tf.gather_nd(bboxes_score, index), [1, -1, 1])
    bboxes = tf.expand_dims(tf.gather_nd(bboxes, index), axis = 0)
    
    dets = K.concatenate(dets, axis = 0)
    detscores = K.concatenate(detscores, axis = 0)
    mask = detscores[:, 0] >= cfg.DETECTION_MIN_CONFIDENCE
    #person only
    class_boxes = tf.boolean_mask(dets, mask)
    class_box_scores = tf.boolean_mask(detscores[:, 0], mask)
    nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, 100, iou_threshold=cfg.DETECTION_NMS_THRESHOLD)
    dets = tf.expand_dims(K.gather(class_boxes, nms_index), axis = 0)
    detscores = K.reshape(K.gather(class_box_scores, nms_index), [1, -1, 1])
    
    return [detscores, dets, bboxes, bboxes_score]

class YOLO:
    def __init__(self, config, *args, **kwargs):
        self.cfg = config
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self,):
        if self.cfg.M in ['yolov3', 'yolov3-tiny']:
            return YOLOv3(self.cfg, *self.args, **self.kwargs)
        elif self.cfg.M in ['yolov4', 'yolov4-tiny']:
            return YOLOv4(self.cfg, *self.args, **self.kwargs)
        else:
            raise ValueError('unsupported model type')

def det_batch(fms, cfg):
    dets, scores = [], []
    for i, feats in enumerate(fms):
        anchors = K.reshape(K.constant(cfg.anchors[i]), 
                            [1, 1, 3, 2])
        
        shape = tf.shape(feats)[1:3]
        
        grid = tf.cast(tf.expand_dims(\
                tf.stack(tf.meshgrid(tf.range(0, shape[1]), 
                tf.range(0, shape[0])), axis = -1), axis = 2), 'float32')
            
        feats = tf.reshape(feats, [-1,
                                   shape[0], 
                                   shape[1], 
                                   cfg.YOLO_NUM_ANCHORS, 
                                   cfg.YOLO_NUM_CLASSES + 5])
        
        conf = tf.sigmoid(feats[..., 4:5])
        prob = tf.sigmoid(feats[..., 5:])
        score = conf * prob
        
        xy = (tf.sigmoid(feats[..., :2]) + grid) /\
            tf.cast(shape[::-1], 'float32')
            
        wh = tf.exp(feats[..., 2:4]) * anchors / \
            tf.cast(tf.constant(cfg.IMAGE_SIZE), 'float32')
            
        yxyx = tf.concat([xy[..., ::-1] - wh[..., ::-1]/2., 
                          xy[..., ::-1] + wh[..., ::-1]/2.], axis = -1)
        
        B = tf.shape(feats)[0]
        yxyx = K.reshape(yxyx, [B, -1, 4])
        batch_index = tf.reshape(tf.range(B), [-1, 1, 1])
        yxyxb = tf.concat([yxyx, 
                           tf.ones_like(yxyx[..., 0]) * batch_index, 
                           ], axis = -1)
        
        yxyxb = tf.reshape(yxyxb, [-1, 5])
        
        score = K.reshape(score, [-1, cfg.YOLO_NUM_CLASSES])
        dets.append(yxyxb)
        scores.append(score)
        
    dets = K.concatenate(dets, axis = 0)
    scores = K.concatenate(scores, axis = 0)
    mask = scores[:, 0] >= cfg.DETECTION_MIN_CONFIDENCE
    #person only
    class_boxes = tf.boolean_mask(dets, mask)
    
    class_box_scores = tf.boolean_mask(scores[:, 0], mask)
    
    nms_index = tf.image.non_max_suppression(
                class_boxes[..., :4], 
                class_box_scores, 
                100, 
                iou_threshold=cfg.DETECTION_NMS_THRESHOLD)
    
    dets = tf.expand_dims(K.gather(class_boxes, nms_index), axis = 0)
    scores = K.reshape(K.gather(class_box_scores, nms_index), [1, -1, 1])
    
    return [dets, scores]

def det(fms, cfg):
        
    dets, scores = [], []
    for i, feats in enumerate(fms):
        anchors = K.reshape(K.constant(cfg.anchors[i]), 
                            [1, 1, 3, 2])
        
        feat = feats[0, ...]
        shape = tf.shape(feat)[:2]
        grid = tf.cast(tf.expand_dims(\
                tf.stack(tf.meshgrid(tf.range(0, shape[1]), 
                tf.range(0, shape[0])), axis = -1), axis = 2), 'float32')
            
        feat = tf.reshape(feat, [shape[0], 
                                shape[1], 
                                cfg.YOLO_NUM_ANCHORS, 
                                cfg.YOLO_NUM_CLASSES + 5])
        
        conf = tf.sigmoid(feat[..., 4:5])
        prob = tf.sigmoid(feat[..., 5:])
        score = conf * prob
        
        xy = (tf.sigmoid(feat[..., :2]) + grid) /\
            tf.cast(shape[::-1], 'float32')
            
        wh = tf.exp(feat[..., 2:4]) * anchors / \
            tf.cast(tf.constant(cfg.IMAGE_SIZE), 'float32')
            
        yxyx = tf.concat([xy[..., ::-1] - wh[..., ::-1]/2., 
                          xy[..., ::-1] + wh[..., ::-1]/2.], axis = -1)
        
        yxyx = K.reshape(yxyx, [-1, 4])
        
        score = K.reshape(score, [-1, cfg.YOLO_NUM_CLASSES])
        dets.append(yxyx)
        scores.append(score)
        
    dets = K.concatenate(dets, axis = 0)
    scores = K.concatenate(scores, axis = 0)
    mask = scores[:, 0] >= cfg.DETECTION_MIN_CONFIDENCE
    #person only
    class_boxes = tf.boolean_mask(dets, mask)
    
    class_box_scores = tf.boolean_mask(scores[:, 0], mask)
    
    nms_index = tf.image.non_max_suppression(class_boxes, 
                                             class_box_scores, 
                                             100, 
                                             iou_threshold=cfg.DETECTION_NMS_THRESHOLD)
    
    dets = tf.expand_dims(K.gather(class_boxes, nms_index), axis = 0)
    scores = K.reshape(K.gather(class_box_scores, nms_index), [1, -1, 1])
    
    return [dets, scores]