#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 00:45:19 2021

@author: hu
"""
import cv2
import numpy as np
import random
class Package:
    '''
    this class provides some basic image processing information.
    package is the only element passing between various of tasks (detection/tracking/web interface)
    '''
    def __init__(self, 
                 _image : np.ndarray, 
                 _timestamp : float, 
                 _id : int,):
        self.image = _image         #image
        self.timestamp = _timestamp #the time of getting (reading) image
        self.id = _id               #the id of image source (camera id)
        self.detection = None       #a dict of detection results
        self.prediction = None      #a list of prediction/interpolation results
        self.filtering = None      #a list of tracking/filtered results
        self._bProcessing = False   #(True) indicates the image is processing by a task
        self._bDetected = False     #(True) indicates the image is detected by a detection network
        self._bPredetected = False  #(True) indicates the image is predicted (no detection)
        self._bTracked = False      #(True) indicates the detection results and tracking results are filtered
        self._bPlayed = False        #(True) indicates the processed image was send to the web interface
        self._bReadyToPlay = False  #(True) indicates this package is ready to play
    
    
    
    def play(self,):
        if not self._bReadyToPlay: return False
        if self._bTracked:
            return self.drawItems(self.filtering), self.timestamp
        elif self._bPredicted:
            return self.drawItems(self.prediction), self.timestamp
        self._pPlayed = True
        raise ValueError('something wrong')
             
    
    def drawItems(self, items, *args, **kwargs):
        img = self.image.copy()
        for _box, _id, _color in items:
            #draw bounding box
            _c = (192, 192, 192) if _color is None else _color
            cx, cy, w, h = _box
            x, y, w, h = int(cx - 0.5 * w), int(cy - 0.5 * h), int(w), int(h)
            img = cv2.rectangle(img, 
                                (x, y), 
                                (x + w, y + h), 
                                (int(_c[0]), int(_c[1]), int(_c[2])), 
                                2)
            
            #draw head with text
            img = cv2.rectangle(img, 
                                (x, y-20), 
                                (x+w, y),
                                (int(_c[0]), int(_c[1]), int(_c[2])), 
                                -1)
            
            img = cv2.putText(img, 
                              '#ID%d'%_id, 
                              (int(x), int(y - 7)), 
                              cv2.FONT_HERSHEY_TRIPLEX,
                              0.5, 
                              (255, 255, 255),
                              1)
        return img
        
        
    def setFrameID(self, frame_id):
        self.frame_id = frame_id
        return self
    
    def getImage(self,):
        return self.image
    
    def getTimestamp(self,):
        return self.timestamp
    
    def getID(self,):
        return self.id
    
    def setStates(self, 
                  _bProcessing = None,
                  _bDetected = None,
                  _bPredetected = None,
                  _bTracked = None):
        if _bProcessing is not None: self._bProcessing = _bProcessing
        if _bDetected is not None: self._bDetected = _bDetected
        if _bPredetected is not None : self._bPredetected = _bPredetected
        if _bTracked is not None : self._bTracked = _bTracked
        
    def getStates(self,):
        return self._bProcessing, self._bDetected, self._bPredetected, self._bTracked
        

class PackageList:
    '''
    this class manage to provide smooth detection and tracking results.
    [pkg_1, pkg_2, ..., pkg_n]. not all packages will be detected or tracked.
    suppose pkg_1 and pkg_i are detected and tracked. this class is used to fill
    the information between pkg_1 and pkg_i. so that to providing smooth detection/tracking
    effects.
    '''
    def __init__(self,):
        self.frame_id = -1
        self.packageList = []
        self.playList = []
        
    def stackPackage(self, package):
        self.frame_id += 1
        self.packageList.append(package.setFrameID(self.frame_id))
        self._tide_packageList()
    
        if len(self.packageList) > 20:
            print('Warning: too many [>20] packages stacked [camera id {}]'.format(package.getID()))
    
    def _tide_packageList(self,):
        '''
        function: clean up packages, incase too many packages stacked in the package list.
        solution: detete some old packages.
        '''
        _bTide = False
        invalidItems = []
        for i,  pkg in enumerate(self.packageList):
            if not pkg.getStates()[0]:
                invalidItems.append(i)
                continue
            _bTide = True
            break
        
        if _bTide:
            for i in invalidItems[::-1]:del self.packageList[i]
            
    def _tide_playList(self,):
        '''
        function: clean up packages, in case too many packages stacked in the play list.
        solution: randomly delete some packages
        '''
        delete_set = set()
        if len(self.playList) > 20:
            while len(delete_set) < (len(self.playList) - 20):
                delete_set.add(random.randint(0, len(self.playList)-1))
                
        self.playList = [pkg for i, pkg in enumerate(self.playList) if i not in delete_set]
            
    def getValidDetectionPackage(self,):
        if len(self.packageList) == 0: return None
        if self.packageList[-1].getStates()[0]:return None
        pkg = self.packageList[-1]
        pkg.setStates(True)
        return pkg
    
    
    def getValidTrackingPackage(self,):
        if len(self.packageList) == 0: return None
        for pkg in self.packageList:
            if pkg.getStates()[1] and not pkg.getStates()[3]:
                return pkg
        return None
    
    
    def fillUpPackages(self, tracker):
        '''
        object detection is much slower than image reading speed.
        in order to give smooth tracking results, those images that
        are not been detected or tracked will be fill up with detection and tracking 
        results by using interpolation-like method.
        '''
        self._tide_packageList()
        index = -1
        for i, pkg in enumerate(self.packageList[1:]):
            if pkg.getStates()[0]:
                index = i
                break
            
        if index <= -1:return
        
        #states = [pkg.getStates()[0] for pkg in self.packageList]
        #frames = [pkg.frame_id for pkg in self.packageList]
        #print('fillUpPackagaes: {}\n{}\n{}'.format(index, states, frames))
        
        self.packageList[0]._bReadyToPlay = True
        for i in range(1, index):
            pkg = self.packageList[i]
            pkg.prediction = tracker.getInterpolation(pkg.timestamp, 'linear')
            pkg._bPredicted = True
            pkg._bReadyToPlay = True
        self.packageList[index + 1]._bReadyToPlay = True
            
            
        self.playList += self.packageList[:(index+1)]
        self.packageList = self.packageList[(index+1):]
        
        self._tide_playList()
    
            
    def play(self,):
        while True:
            if len(self.playList) == 0:
                return False
            _pkg = self.playList.pop(0)
            return _pkg.play()
            
        
        #fill up the gap between two tracking points
        