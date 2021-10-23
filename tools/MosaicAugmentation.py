#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:47:50 2019

@author: hu
data augmentation
"""
import numpy as np
import cv2
from random import shuffle
from numpy import random
class DataAugmentation():
    def __init__(self, mode, cfg):
        assert mode in ['training', 'validation']
        self.config = cfg
        self.mode = mode
    
    def __call__(self, images, bboxes, blocks):
        '''
        image: BGR
        bbox: xywh
        '''
        
        self.blocks = blocks[0] + blocks[1] + blocks[2] + blocks[3]
        self.images = images
        self.bboxes = [np.array(boxes, dtype = np.float32) for boxes in bboxes]
        self.mean = self.config.MEAN_PIXEL
        if self.mode == 'training':
            return self.augmentation()
        elif self.mode == 'validation':
            return self.augmentation_val()
        else:
            raise ValueError('unsupported type...')
    
    def bgDim(self,):
        for i, I in enumerate(self.images):
            if 1:#random.randint(2):
                self.bboxes[i] = self.bboxes[i][:, :-1]
                continue
            B = self.bboxes[i]
            border = 15
            ih, iw, _ = I.shape
            new_I = cv2.blur(I.copy(), (border * 2, border * 2))
            for b in B:
                x, y, w, h = b[:4].copy()
                x11, y11, x12, y12 = int(x), int(y), int(x + w), int(y + h)
                x21, y21, x22, y22 = int(x11 - border), int(y11 - border), int(x12 + border), int(y12 + border)
                x11 = 0 if x11 < 0 else x11
                y11 = 0 if y11 < 0 else y11
                x12 = iw - 1 if x12 >= iw else x12
                y12 = ih - 1 if y12 >= ih else y12
                x21 = 0 if x21 < 0 else x21
                y21 = 0 if y21 < 0 else y21
                x22 = iw - 1 if x22 >= iw else x22
                y22 = ih - 1 if y22 >= ih else y22
                #border_block = cv2.blur(I[y21:y22, x21:x22, :], (border * 2, border * 2))
                #new_I[y21:y22, x21:x22, :] = border_block
                new_I[y11:y12, x11:x12, :] = I[y11:y12, x11:x12, :]
            self.images[i] = new_I
            self.bboxes[i] = B[:, :-1]

    def merge_blocks(self, blocks):
        blocks.sort(key = lambda x : x[1][0] * x[1][1])
        new_blocks = []
        pair_block = []
        while len(blocks) > 1:
            pair_block = [blocks.pop(0), blocks.pop(0)]
            w1, h1, w2, h2 = edge_length = pair_block[0][1] + pair_block[1][1]
            min_edge = np.argmin(np.array([w1 + w2, h1 + h2]))
            if min_edge == 0: #width
                block_w, block_h = edge_length[0] + edge_length[2], max(edge_length[1], edge_length[3])
                block_img = np.zeros([block_h, block_w, 3], dtype = np.uint8)
                block_img[:h1, :w1] = pair_block[0][0]
                block_img[:h2, w1:] = pair_block[1][0]
                pair_block[1][2][:, [0, 2]] += w1
                block_boxes = np.concatenate([pair_block[0][2], pair_block[1][2]], axis = 0)
                new_blocks.append([block_img, [block_w, block_h], block_boxes])
            else:#h
                block_w, block_h = max(edge_length[0], edge_length[2]), edge_length[1] + edge_length[3]
                block_img = np.zeros([block_h, block_w, 3], dtype = np.uint8)
                block_img[:h1, :w1] = pair_block[0][0]
                block_img[h1:, :w2] = pair_block[1][0]
                pair_block[1][2][:, [1, 3]] += h1
                block_boxes = np.concatenate([pair_block[0][2], pair_block[1][2]], axis = 0)
                new_blocks.append([block_img, [block_w, block_h], block_boxes])
        if len(blocks) > 0:
            new_blocks += blocks
        return new_blocks
        
    def mosaic(self,):
        blocks = self.blocks
        for i, I in enumerate(self.images):
            boxes = self.bboxes[i]
            ih, iw, _ = I.shape
            for box in boxes:
                x, y, w, h = box[:4].copy()
                border = w//2
                if border < 20:border = 20
                x11, y11, x12, y12 = int(x), int(y), int(x + w), int(y + h)
                x21, y21, x22, y22 = int(x11 - border), int(y11 - border), int(x12 + border), int(y12 + border)
                x11 = 0 if x11 < 0 else x11
                y11 = 0 if y11 < 0 else y11
                x12 = iw - 1 if x12 >= iw else x12
                y12 = ih - 1 if y12 >= ih else y12
                x21 = 0 if x21 < 0 else x21
                y21 = 0 if y21 < 0 else y21
                x22 = iw - 1 if x22 >= iw else x22
                y22 = ih - 1 if y22 >= ih else y22
                block = [I[y21:y22, x21:x22, :], [x22 - x21, y22 - y21], np.array([[x11 - x21, y11 - y21, x12 - x21, y12 - y21] + list(box[4:])])]
                blocks.append(block)
        #merge blocks
        while len(blocks) > 1:
            blocks = self.merge_blocks(blocks)
        
        pw, ph = blocks[0][1]
        me = max((pw + 1)//2, (ph+1)//2)
        bk_imgs = [cv2.resize(image, dsize = (int(image.shape[1] * (me/min(image.shape[:2]))) + 1, int(image.shape[0] * (me/min(image.shape[:2]))) + 1), interpolation = cv2.INTER_LINEAR)[:me, :me] if min(image.shape[:2]) < me else image[:me, :me] for image in self.images]
        bk_img = np.zeros([me * 2, me * 2, 3], dtype = np.uint8)
        bk_img[:me, :me, :], bk_img[-me:, :me, :], bk_img[:me, -me:, :], bk_img[-me:, -me:, :] = bk_imgs
        bk_img[:ph, :pw] *= np.expand_dims(np.where(blocks[0][0].sum(axis = -1) == 0, 1, 0), axis = -1).astype(np.uint8)
        bk_img[:ph, :pw] += blocks[0][0]
        
        blocks[0][0] = bk_img
        self.images = [blocks[0][0]]
        self.bboxes = [blocks[0][2]]

    def augmentation_val(self,):
        self.bbox = [boxes.reshape([-1, 4]) for boxes in self.bboxes]
        self.xywh2xyxy()
        self.mold_box()
        self.convertFromInts()
        self.convertColorBGR2RGB()
        self.mold()
        self.normalizeCoordinate()
        self.xyxy2yxyx()
        self.ids = [np.ones([b.shape[0], 1]) for b in self.bbox]
        return self.image[0], self.bbox[0], self.ids[0], self.meta
    
    def augmentation(self,):
        self.mosaic()
        self.bgDim()
        self.mold_box()
        self.convertFromInts()
        if self.mode == 'training':
            self.photometricDistort()
            self.randomMirror()

        self.mold()
        self.normalizeCoordinate()
        self.xyxy2yxyx()
        self.boxPadding()
        self.createAssociationMatrix()
        self.boxRemoveIndex()
        return self.images[0], self.bboxes[0], self.indices[0], self.locations[0]
    
    def mold_box(self,):
        for i, image in enumerate(self.images):
            ih, iw, _ = image.shape
            boxes = self.bboxes[i]
            boxes[:, 0][boxes[:, 0] < 0.0] = 0
            boxes[:, 1][boxes[:, 1] < 0.0] = 0
            boxes[:, 2][boxes[:, 2] < 0.0] = 0
            boxes[:, 3][boxes[:, 3] < 0.0] = 0
            boxes[:, 0][boxes[:, 0] > iw] = iw
            boxes[:, 2][boxes[:, 2] > iw] = iw
            boxes[:, 1][boxes[:, 1] > ih] = ih
            boxes[:, 3][boxes[:, 3] > ih] = ih
            self.bboxes[i] = boxes
    
    def boxPadding(self,):
        for i in range(len(self.images)):
            if self.bboxes[i].size == 0:
                self.bboxes[i] = np.ones((self.config.MAX_TRACKING_VOLUME, 6)) * -1
                continue
            boxes = self.bboxes[i]
            t_boxes = boxes.copy()
            t_boxes[:, 4:] = -1
            while len(boxes) < self.config.MAX_TRACKING_VOLUME:
                cl = (self.config.MAX_TRACKING_VOLUME - len(boxes))
                if cl <= len(t_boxes):
                    boxes = np.concatenate((boxes, t_boxes[:cl, :]), axis = 0)
                else:
                    boxes = np.concatenate((boxes, t_boxes), axis = 0)
            boxes = boxes[:self.config.MAX_TRACKING_VOLUME, :]
            self.bboxes[i] = boxes

    def boxRemoveIndex(self,):
        for i in range(len(self.images)):
            if self.bboxes[i].size == 0:
                self.bboxes[i] = np.zeros((self.config.MAX_TRACKING_VOLUME, 4))
                continue
            self.bboxes[i] = self.bboxes[i][:, :4]

    def createAssociationMatrix(self,):
        self.indices = []
        self.locations = []
        for i in range(len(self.images)):
            index = self.bboxes[i][:, -2].reshape([-1, 1])
            location = self.bboxes[i][:, -1].reshape([-1, 1])
            self.indices.append(index)
            self.locations.append(location)
    
    def mold(self,):
        self.metas = []
        self.windows = []
        for i in range(len(self.images)):
            image = self.images[i]
            bbox = self.bboxes[i]
            
            if self.config.M == 'mrcnn':
                image, window, scale = self.mrcnn_resize_image(image)
            elif self.config.M == 'yolov3' or self.config.M == 'yolov4':
                image, window, scale = self.yolo_resize_image(image)
            else:
                raise ValueError('unsupported image size...')
            bbox[:, :4] *= scale
            bbox[:, 0] += window[1]
            bbox[:, 2] += window[1]
            bbox[:, 1] += window[0]
            bbox[:, 3] += window[0]
            self.images[i] = image
            self.windows.append(window)
            self.metas.append(self.compose_image_meta(
                0, image.shape, image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32)))

    def yolo_resize_image(self, image):
        size = self.config.IMAGE_MAX_DIM
        h, w = image.shape[:2]
        image_max = max(h, w)
        scale = size / image_max
        image = cv2.resize(image, dsize = (round(w * scale), round(h * scale)), interpolation = cv2.INTER_CUBIC)
        h, w = image.shape[:2]
        top_pad = (size - h) // 2
        bottom_pad = size - h - top_pad
        left_pad = (size - w) // 2
        right_pad = size - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=128)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        return image[:, :, ::-1].astype(np.float32)/255., window, scale

    def mrcnn_resize_image(self, image):
        size = self.config.IMAGE_MAX_DIM
        h, w = image.shape[:2]
        image_max = max(h, w)
        scale = size / image_max
        image = cv2.resize(image, dsize = (round(w * scale), round(h * scale)), interpolation = cv2.INTER_LINEAR)
        h, w = image.shape[:2]
        top_pad = (size - h) // 2
        bottom_pad = size - h - top_pad
        left_pad = (size - w) // 2
        right_pad = size - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        return image.astype(np.float32)-self.mean, window, scale
    
    def normalizeCoordinate(self,):
        for i in range(len(self.images)):
            ih, iw, _ = self.images[i].shape
            if self.bboxes[i].size == 0:continue
            self.bboxes[i][:, 0] /= iw
            self.bboxes[i][:, 1] /= ih
            self.bboxes[i][:, 2] /= iw
            self.bboxes[i][:, 3] /= ih
    
    def randomMirror(self,):
        if random.randint(2):
            for i in range(len(self.images)):
                _, width, _ = self.images[i].shape
                self.images[i] = self.images[i][:, ::-1, :]
                self.bboxes[i][:, 0] = width - self.bboxes[i][:, 0]
                self.bboxes[i][:, 2] = width - self.bboxes[i][:, 2]
                self.bboxes[i][:, :4] = self.bboxes[i][:, [2, 1, 0, 3]]
                
    def compose_image_meta(self, image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
        meta = np.array(
            [image_id] +                  # size=1
            list(original_image_shape) +  # size=3
            list(image_shape) +           # size=3
            list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
            [scale] +                     # size=1
            list(active_class_ids)        # size=num_classes
        )
        return meta
    
    def xyxy2yxyx(self,):
        for i in range(len(self.images)):
            if self.bboxes[i].size == 0:continue
            self.bboxes[i][:, :4] = self.bboxes[i][:, [1, 0, 3, 2]]
    
    def xywh2xyxy(self,):
        for i in range(len(self.images)):
            if self.bboxes[i].size == 0:continue
            self.bboxes[i][:, 2 : 4] = self.bboxes[i][:, :2] + self.bboxes[i][:, 2 : 4]

    def xyxy2xywh(self,):
        for i in range(len(self.images)):
            if self.bboxes[i].size == 0:continue
            self.bboxes[i][:, 2 : 4] = self.bboxes[i][:, 2 : 4] - self.bboxes[i][:, :2]
    
    def photometricDistort(self,):
        self.convertColorBGR2HSV()
        self.randomBrightness()
        self.convertColorHSV2RGB()
    
    def convertFromInts(self,):
        for i in range(len(self.images)):
            self.images[i] = self.images[i].astype(np.float32)
       
    def randomBrightness(self,):
        for i in range(len(self.images)):
            if random.randint(2):continue
            rand = random.uniform(0.7, 1.3)
            self.images[i][:, :, 2] = self.images[i][:, :, 2] * rand
            
    def convertColorBGR2HSV(self,):
        for i in range(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2HSV)
    def convertColorRGB2HSV(self,):
        for i in range(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_RGB2HSV)
    def convertColorRGB2BGR(self,):
        for i in range(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_RGB2BGR)
    def convertColorBGR2RGB(self,):
        for i in range(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB)
    def convertColorHSV2RGB(self,):
        for i in range(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_HSV2RGB)
    def convertColorHSV2BGR(self,):
        for i in range(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_HSV2BGR)
        


