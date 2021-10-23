#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import List
import cv2
import threading
import time
import numpy as np
import datetime
import _thread
import queue
GLOBAL_TIME_BETWEEN_COMMAND = 0.1
TIME_BEFORE_LAND = datetime.timedelta(seconds=12) # Doit etre superieur à 10 ?

class Videos(threading.Thread):
    '''get files from existed videos
    '''
    def __init__(self, _dirs_dict,):
        '''
        _dirs_dict: {'camid1': [video_path, tiemstamp_path],
                     'camid2': [video_path, timestamp_path]
                     }
        '''
        threading.Thread.__init__(self)
        self.__dirs_dict = _dirs_dict
        self.__image_dict = {}
        self.__caps = {}
        self.__image_queues = dict(zip(list(self.__dirs_dict),
                                   [queue.Queue(1) for _ in self.__dirs_dict]))
        self._parse()
            
    def _parse(self,):
        self.__timestamp_list = []
        
        for _camId in self.__dirs_dict:
            video_path, timestamp_path = self.__dirs_dict[_camId]
            assert os.path.exists(video_path)
            assert os.path.exists(timestamp_path)
            #timestamp
            _timestamp = open(timestamp_path)
            _lines = _timestamp.readlines()
            for j, _line in enumerate(_lines):
                self.__timestamp_list.append([float(_line.strip()), [_camId, j]])
            self.__timestamp_list.sort(key = lambda x : x[0])
            self.__init_timestamp = self.__timestamp_list[0][0]
            #video
            _video = cv2.VideoCapture(video_path)
            self.__caps[_camId] = _video
            
            
    def run(self,):
        _start = time.time()
        _previous_time = time.time()
        _previous_timestamp = self.__init_timestamp
        while len(self.__timestamp_list):
            _timestamp, [_camId, _frame_index] = self.__timestamp_list.pop(0)
            self.__caps[_camId].set(cv2.CAP_PROP_POS_FRAMES, _frame_index)
            _success, _image = self.__caps[_camId].read()
            if not _success:continue
            #play video with real framerate
            while _timestamp - self.__init_timestamp > time.time() - _start:
                time.sleep(0.005)
            _current_time = time.time()
            while _timestamp - _previous_timestamp > _current_time - _previous_time:
                _current_time = time.time()
                time.sleep(0.001)
            _previous_timestamp = _timestamp
            _previous_time = _current_time
            self.__image_queues[_camId].put([True, _image, _timestamp])
            
    def getShape(self, _camId):
        return [self.__cap[_camId].get(4), self.__cap[_camId].get(3), 3]
    
    def read(self, _camId):
        return self.__image_queues[_camId].get()
    
class VideoSource:
    def __init__(self, 
                 _videos : Videos,
                 _camId):
        self.__ownedCameraId = _camId
        self.__videos = _videos
    
    def read(self,):
        return self.__videos.read(self.__ownedCameraId)
    
    def shape(self,):
        return self.__videos.getShape(self.__ownedCameraId)
    
    
class Reader:
    knownCameras = {}
    def __init__(self, 
                 _source,
                 _sourceId,
                 _isTriggered = False,
                 _altName = None,
                 _type : str = None,
                 _queueSize = 3,
                 _bThreading = True, 
                 _bRecording = False,
                 _maxsize = 1):
        self.__source = _source
        self.__sourceId = _sourceId
        Reader.knownCameras[_sourceId] = self
        self.__altName = _altName
        self.__buffer = queue.Queue(_queueSize)
        self.__source_type = None
        self.__isTriggered = _isTriggered
        self.__pause = False if self.__isTriggered else True
        self.__source_type = None
        self.__bThreading = _bThreading
        self.__bThreading_start = False
        self.__bRecording = _bRecording
        self.__bRecording_start = False
        self.__type = _type
        
        self._parse()
        self.__buffer = queue.Queue(10)
        self.__image_queue = queue.Queue(_maxsize)
        self.__data = {}
        self.__dataLock =  threading.Lock()
        
        if self.__bThreading and self.__bRecording:
            try:
                _thread.start_new_thread(self._thread_record, ())
            except:
                raise ValueError('unable to create thread')
                
        if self.__bThreading:
            try:
                _thread.start_new_thread(self._thread_read, ())
            except:
                raise ValueError('unable to create thread')
    
    def isDrone(self):
        return self.__type == 'drone'
    
    def getData(self,key):
        res = None
        self.__dataLock.acquire()
        if key in self.__data:
            res = self.__data[key]
        self.__dataLock.release()
        return res

    def putData(self,key, value):
        self.__dataLock.acquire()
        self.__data[key] = value
        self.__dataLock.release()

    def getDataKeys(self):
        self.__dataLock.acquire()
        res = self.__data.keys()
        self.__dataLock.release()
        return res
    
    def __del__(self,):
        self.__bThreading = False
        self.__bRecording = False
        if hasattr(self, '__video_writer'):self.__video_writer.release()
        print('close video writer for {}'.format(self.__altName))
        if hasattr(self, '__video_timestamp'):self.__video_timestamp.close()
        print('close video timestamp for {}'.format(self.__altName))
    
    def _write_video(self, img, timestamp):
        assert hasattr(self, '__video_writer')
        assert hasattr(self, '__video_timestamp')
        self.__video_timestamp.write(str(timestamp)+'\n')
        self.__video_timestamp.flush()
        self.__video_writer.write(img)
    
    def getId(self,):
        return self.__sourceId
    
    def pause(self,):
        self.__pause = True
        
    def resume(self,):
        self.__pause = False
        
    def read(self,):
        #_state = False
        while self.__pause:
            img, timestamp = np.array([]), -1
            time.sleep(0.03)
        
        img, timestamp = self.__image_queue.get()
        return True, img, timestamp
        '''
        try:
            img, timestamp = self.__image_queue.get_nowait()
            _state = True
        except queue.Empty:
            img, timestamp = np.array([]), -1
            pass
        print('reader.read:', _state,img, timestamp)
        return _state, img, timestamp'''
        
    def _thread_read(self,):
        self.__bThreading_start = True
        while self.__bThreading:
            if self.__source_type == 'callable':
                success, img, timestamp = self._read()
            else:
                success, img = self._read()[:2]
                timestamp = time.time()
            if not success:
                if self.__source_type == 'stream':
                    self._parse()
                    continue
                else:
                    break
            else:
                if self.__bRecording:
                    self.__buffer.put([img, timestamp])
            
            self.__image_queue.put([img, timestamp])
            '''
            if self.__source_type in ['callable', 'stream']:
                try:
                    self.__image_queue.put_nowait([img, timestamp])
                except queue.Full:
                    self.__image_queue.get_nowait()
                    self.__image_queue.put_nowait([img, timestamp])
                    pass
            else:
                self.__image_queue.put([img, timestamp])
            '''
                
        self.__bThreading_start = False
                
    def _thread_record(self,):
        self.__bRecording_start = True
        date = time.localtime()
        if not os.path.exists('records'):
            os.mkdir('records')
        
        path = os.path.join('records', 
                            '%d-%d-%d-%d-%d-%d-%s'%(date.tm_year,
                                                 date.tm_mon,
                                                 date.tm_mday,
                                                 date.tm_hour,
                                                 date.tm_min,
                                                 date.tm_sec,
                                                 self.source_type \
                                                     if self.__altName is None else\
                                                         self.__altName))
        i = 1
        while os.path.exists(path):
            path += '-%d'%i
            i+= 1
            
        os.mkdir(path)
        ih, iw = self.__shape[:2]
        self.__video_writer = cv2.VideoWriter(filename = os.path.join(path, 'output.avi'),
                                            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                            fps = 30, 
                                            frameSize = (int(iw), int(ih))
                                            )
        self.__video_timestamp = open(os.path.join(path, 'timestamp.txt'), 'w')
        
        while self.__bThreading and self.__bRecording:
            if self.__buffer.qsize() > 5:print('video writer is too slow {}'.format(self.buffer.qsize()))
            self._write_video(*self.__buffer.get())
            
        self.__bRecording_start = False
            
    def _parse(self,):
        self.__source_type = None
        if hasattr(self.__source, 'read'):
            self.__cap = self.__source
            self.__source_type = 'callable'
            if hasattr(self.__cap, 'shape'):
                self.__shape = self.__cap.shape
            else:
                success, img = self.__cap.read()[:2]
                if not success:
                    raise ValueError('[Error] Get image failed!')
                self.__shape = img.shape
        else:
            if hasattr(self, '__cap'):self.__cap.release()
            cap = cv2.VideoCapture(self.__source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                self.__cap = cap
                self.__shape = [int(cap.get(4)), int(cap.get(3)), 3]
            else:
                raise ValueError('[Error] Open video [%s] failed!'%self.__source)
            self.__source_type = 'stream'
        return True
    
    def _open(self,):
        return self._parse()
    
    def _read(self,):
        if self.__source_type == None:
            return False, None
        else:
            return self.__cap.read()


