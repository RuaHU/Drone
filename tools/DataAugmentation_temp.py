#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:47:50 2019

@author: hu
data augmentation
"""
import numpy as np
import cv2
from numpy import random

def draw(image, boxes, name = 'img'):
    image = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        image = cv2.rectangle(image, 
                              (int(x1), int(y1)), 
                              (int(x2), int(y2)),
                              (255, 255, 255))
        
    cv2.imshow(name, image)
    cv2.waitKey(0)

class DataAugmentation():
    def __init__(self, mode, cfg):
        assert mode in ['training', 'validation']
        self.config = cfg
        self.mode = mode
    
    def __call__(self, image1, bbox1):
        '''
        image: BGR
        bbox: xywh
        '''
        self.image = [image1]
        self.bbox = [np.array(bbox1, dtype = np.float32)]
        self.mean = self.config.MEAN_PIXEL
        if self.mode == 'training':
            return self.augmentation()
        elif self.mode == 'validation':
            return self.augmentation_val()
        else:
            raise ValueError('unsupported type...')
    
    def bgDim(self,):
        for i, I in enumerate(self.image):
            if 1:#random.randint(2):
                self.bbox[i] = self.bbox[i][:, :-1]
                continue
            B = self.bbox[i]
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
            self.image[i] = new_I
            self.bbox[i] = B[:, :-1]

    def augmentation_val(self,):
        self.bbox = [b.reshape([-1, 4]) for b in self.bbox]
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
        self.xywh2xyxy()
        #draw(self.image[0], self.bbox[0], name = 'img1')
        if self.mode == 'training':
            self.randomBoxNoise()
            self.random_crop()
        self.xyxy2xywh()
        self.bgDim()
        self.xywh2xyxy()
        self.mold_box()
        #draw(self.image[0], self.bbox[0], name = 'img2')
        #return
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
        return self.image[0], self.bbox[0], self.ids1, self.loc1
    
    def mold_box(self,):
        for i, boxes in enumerate(self.bbox):
            ih, iw = self.image[i].shape[:2]
            boxes = boxes[:, :4]
            s = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            np.clip(boxes, 0, 1e10, out = boxes)
            np.clip(boxes[:, ::2], 0, iw, out = boxes[:, ::2])
            np.clip(boxes[:, 1::2], 0, ih, out = boxes[:, 1::2])
            ns = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            self.bbox[i] = self.bbox[i][ns > 0.6 * s]
    
    def random_crop(self,):
        if random.randint(2):
            ih, iw = self.image[0].shape[:2]
            if random.randint(2):
                s = self.image[0].shape[:2]
                b = random.uniform(0, 0.25, 4)
                b[2:] = 1 - b[2:]
                b = (b * np.array([*s, *s]))
            else:
                minx, maxx = self.bbox[0][:, [0, 2]].min(), \
                             self.bbox[0][:, [0, 2]].max()
                miny, maxy = self.bbox[0][:, [1, 3]].min(), \
                             self.bbox[0][:, [1, 3]].max()
                
                minx = np.clip(minx, 0, iw)
                miny = np.clip(miny, 0, ih)
                maxx = np.clip(maxx, 0, iw)
                maxy = np.clip(maxy, 0, ih)
                
                if random.randint(2):
                    b = np.array([random.randint(0, miny),
                                 random.randint(0, minx),
                                 random.randint(maxy, ih),
                                 random.randint(maxx, iw)])
                    
                else:
                    b = np.array([miny, minx, maxy, maxx])
                    b[:2] = b[:2] - random.randint(0, 50, 2)
                    b[2:] = b[2:] + random.randint(0, 50, 2)
                
            np.clip(b[::2], 0, ih, out = b[::2])
            np.clip(b[1::2], 0, iw, out = b[1::2])
            b = b.astype(np.int32)
            self.image[0] = self.image[0][b[0]:b[2], b[1]:b[3], :]
            self.bbox[0][:, :2] -= np.array(b[1::-1])
            self.bbox[0][:, 2:4] -= np.array(b[1::-1])
                
        
    
        
    def boxPadding(self,):
        for i in range(1):
            if self.bbox[i].size == 0:
                self.bbox[i] = np.ones((self.config.MAX_TRACKING_VOLUME, 6)) * -1
                continue
            boxes = self.bbox[i]
            t_boxes = boxes.copy()
            t_boxes[:, 4:] = -1
            while len(boxes) < self.config.MAX_TRACKING_VOLUME:
                cl = (self.config.MAX_TRACKING_VOLUME - len(boxes))
                if cl <= len(t_boxes):
                    boxes = np.concatenate((boxes, t_boxes[:cl, :]), axis = 0)
                else:
                    boxes = np.concatenate((boxes, t_boxes), axis = 0)
            self.bbox[i] = boxes

    def boxRemoveIndex(self,):
        for i in range(1):
            if self.bbox[i].size == 0:
                self.bbox[i] = np.zeros((self.config.MAX_TRACKING_VOLUME, 4))
                continue
            self.bbox[i] = self.bbox[i][:, :4]

    def createAssociationMatrix(self,):
        index1 = self.bbox[0][:, -2].reshape([-1, 1])
        location1 = self.bbox[0][:, -1].reshape([-1, 1])
        self.ids1 = index1
        self.loc1 = location1

    def unmold(self, bboxes, meta):
        ih, iw = meta[0][4:6]
        window = meta[0][7:11]
        if self.config.M in ['dla_34', 'dla_34_new']:
            ratio = meta[0][11]
            dh, dw = window[:2]
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] * ih - dh)/ratio
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] * iw - dw)/ratio
            bboxes[:, 2:] -= bboxes[:, :2]
            return bboxes[:, [1, 0, 3, 2]]
        if self.config.letter_box == True:
            scale_x = scale_y = meta[0][11]
        else:
            scale_x, scale_y = meta[0][11], meta[0][12]
        bboxes[:, [0, 2]] *= ih
        bboxes[:, [1, 3]] *= iw
        bboxes[:, [0, 2]] -= window[0]
        bboxes[:, [1, 3]] -= window[1]
        bboxes[:, [0, 2]] /= scale_y
        bboxes[:, [1, 3]] /= scale_x
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes[:, [1, 0, 3, 2]]
        return bboxes

    def mold(self,):
        self.meta = []
        self.window = []
        for i in range(1):
            image = self.image[i]
            bbox = self.bbox[i]
            
            if self.config.M == 'mrcnn':
                image, window, scale = self.mrcnn_resize_image(image)
                scale_x = scale_y = scale[0]
            elif self.config.M in ['dla_34', 'dla_34_new']:
                image, window, scale = self.dla_resize_image(image)
                scale_x = scale_y = scale[0]
            elif self.config.M in self.config.backbones:
                image, window, scale = self.backbone_resize(image)
                scale_x = scale_y = scale[0]
            elif hasattr(self.config, 'letter_box'):
                if self.config.letter_box == False:
                    image, window, scale = self.yolo_resize_image_v4(image)
                    scale_x, scale_y = scale
                elif self.config.letter_box == True:
                    image, window, scale = self.yolo_resize_image_v3(image)
                    scale_x = scale_y = scale[0]
            else:
                raise ValueError('unsupported type...')
            bbox[:, 0] *= scale_x
            bbox[:, 0] += window[1]
            bbox[:, 2] *= scale_x
            bbox[:, 2] += window[1]
            bbox[:, 1] *= scale_y
            bbox[:, 1] += window[0]
            bbox[:, 3] *= scale_y
            bbox[:, 3] += window[0]
            self.image[i] = image
            self.window.append(window)
            self.meta.append(self.compose_image_meta(
                0, image.shape, image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32)))

    def backbone_resize(self, image):
        image = self.config.net_dict[self.config.M][1](image, data_format="channels_last")
        h, w = image.shape[:2]
        image_max = max(h, w)
        size = 608
        scale = size / image_max
        image = cv2.resize(image, dsize = (round(w*scale), round(h*scale)), interpolation = cv2.INTER_CUBIC)
        h, w = image.shape[:2]
        top_pad = (size - h) // 2
        bottom_pad = size - h - top_pad
        left_pad = (size - w) // 2
        right_pad = size - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        return image, window, [scale]

    def dla_resize_image(self, image):
        shape = image.shape[:2]
        height, width = self.config.IMAGE_SIZE
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.config.FILL_COLOR)  # padded rectangular
        window = (top, left, dh, dw)
        return img/255., window, [ratio]

    def yolo_resize_image_v3(self, image):
        if hasattr(self.config, 'RANDOM_SIZE') and self.mode == 'training':
            size = self.config.RANDOM_SIZE[random.randint(len(self.config.RANDOM_SIZE))]
        else:
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
        return image/255., window, [scale]
    
    def yolo_resize_image_v4(self, image):
        if hasattr(self.config, 'RANDOM_SIZE') and self.mode == 'training':
            size = self.config.RANDOM_SIZE[random.randint(len(self.config.RANDOM_SIZE))]
        else:
            size = self.config.IMAGE_MAX_DIM
        h, w = image.shape[:2]
        scale_x = size / w
        scale_y = size / h
        image = cv2.resize(image, None, None, fx = scale_x, fy = scale_y, interpolation = cv2.INTER_CUBIC)
        window = (0, 0, image.shape[0], image.shape[1])
        return image/255., window, [scale_x, scale_y]

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
        return image.astype(np.float32)-self.mean, window, [scale]
    
    def normalizeCoordinate(self,):
        for i in range(1):
            if hasattr(self, 'image'):
                height, width, _ = self.image[i].shape
            else:
                height, width = self.height, self.width
            if self.bbox[i].size == 0:
                continue
            self.bbox[i][:, 0] /= width
            self.bbox[i][:, 1] /= height
            self.bbox[i][:, 2] /= width
            self.bbox[i][:, 3] /= height
    
    
    def randomBoxNoise(self,):
        sigma = self.config.MAX_OFFSET_LEN
        for boxes in self.bbox:
            for box in boxes:
                w, h = (box[2:4] - box[:2]).astype(np.float32)
                rw, rh = np.random.uniform(-sigma, sigma, 2)
                rw1, rh1 = np.random.uniform(-sigma, sigma, 2)
                #ow1, oh1 = np.random.uniform(0, rw, 1)[0], np.random.uniform(0, rh, 1)[0]
                #ow2, oh2 = rw - ow1, rh - oh1
                box[:4] = box[:4] + np.array([w * rw, h * rh, w * rw1, h * rh1])
    
    def randomMirror(self,):
        if random.randint(2):
            for i in range(1):
                _, width, _ = self.image[i].shape
                self.image[i] = self.image[i][:, ::-1, :]
                self.bbox[i][:, 0] = width - self.bbox[i][:, 0]
                self.bbox[i][:, 2] = width - self.bbox[i][:, 2]
                self.bbox[i][:, :4] = self.bbox[i][:, [2, 1, 0, 3]]
                
    def compose_image_meta(self, image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
        meta = np.array(
            [image_id] +                  # size=1
            list(original_image_shape) +  # size=3
            list(image_shape) +           # size=3
            list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
            scale +                     # size=1
            list(active_class_ids)        # size=num_classes
        )
        return meta
    
    def xyxy2yxyx(self,):
        for i in range(1):
            if self.bbox[i].size == 0:
                continue
            self.bbox[i][:, :4] = self.bbox[i][:, [1, 0, 3, 2]]
    
    def xywh2xyxy(self,):
        for i in range(1):
            if self.bbox[i].size == 0:
                continue
            self.bbox[i][:, 2 : 4] = self.bbox[i][:, :2] + self.bbox[i][:, 2 : 4]

    def xyxy2xywh(self,):
        for i in range(1):
            if self.bbox[i].size == 0:
                continue
            self.bbox[i][:, 2 : 4] = self.bbox[i][:, 2 : 4] - self.bbox[i][:, :2]
    
    def photometricDistort(self,):
        self.convertColorBGR2HSV()
        self.randomBrightness()
        self.convertColorHSV2RGB()
    
    def convertFromInts(self,):
        for i in range(1):
            self.image[i] = self.image[i].astype(np.float32)
       
    def randomBrightness(self,):
        if random.randint(2):
            for i in range(1):
                rand = random.uniform(0.7, 1.3)
                self.image[i][:, :, 2] = self.image[i][:, :, 2] * rand
            
    def convertColorBGR2HSV(self,):
        for i in range(1):
            self.image[i] = cv2.cvtColor(self.image[i], cv2.COLOR_BGR2HSV)
    def convertColorRGB2HSV(self,):
        for i in range(1):
            self.image[i] = cv2.cvtColor(self.image[i], cv2.COLOR_RGB2HSV)
    def convertColorRGB2BGR(self,):
        for i in range(1):
            self.image[i] = cv2.cvtColor(self.image[i], cv2.COLOR_RGB2BGR)
    def convertColorBGR2RGB(self,):
        for i in range(1):
            self.image[i] = cv2.cvtColor(self.image[i], cv2.COLOR_BGR2RGB)
    def convertColorHSV2RGB(self,):
        for i in range(1):
            self.image[i] = cv2.cvtColor(self.image[i], cv2.COLOR_HSV2RGB)
    def convertColorHSV2BGR(self,):
        for i in range(1):
            self.image[i] = cv2.cvtColor(self.image[i], cv2.COLOR_HSV2BGR)
        



if __name__ == '__main__':
    
    from config import Config
    from generator import GENERATORS
    cfg = Config('yolov3')
    gen = GENERATORS(cfg, 
                     CUHK_SYSU = '/home/hu/Projects/MoT/data/dataset-v2',
                     PRW = '/home/hu/Projects/MoT/data/PRW-v16.04.20').CUHK_SYSU_generator('training')
    da = DataAugmentation('training', cfg)
    while True:
        image, boxes = next(gen)
        da(image, boxes)
    