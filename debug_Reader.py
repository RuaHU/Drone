#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import queue
import glob
import time
import threading
import numpy as np

class Reader:
    def __init__(self, source, sourceId, altName = None, queueSize = 2, bufferSize = 2):
        """
        Class to read on a video source.
        Parameters :
        source : Video/Image Source. Allowed sources are described in method
            self.__loadSource
        sourceId : Int
        """
        self.frames = [cv2.imread(f + '.jpg') for f in ['4', '5', '6']]

        self.__source = source
        self.__sourceId = sourceId
        self.__altName = altName
        self.__image_queue = queue.Queue(queueSize)
        self.__bufferSize = bufferSize
        self.__cap = None
        self.__files = None
        #self.__source_type = self.__loadSource(self.__source)
        #self.__shape = self.getShape()
        # try:
        #     self.__threading = True
        #     #self.runningThreadId = _thread.start_new_thread(self.__thread_read, ())
        #     self.runningThreadId = threading.Thread(target=self.__thread_read, args=(),daemon=True)
        #     self.runningThreadId.start()
        # except:
        #     raise ValueError('unable to create thread')

    def getReaderCamId(self):
        return self.__sourceId

    def __str__(self):
        if self.__altName != None:
            return "Reader with id (%s) for camera (%s)\n" % (self.__sourceId, self.__altName)
        else:
            return "Reader with id (%s)\n" % (self.__sourceId)

    def __del__(self,):
        """
        Ask to close background reading thread.
        """
        print('end')
        # if self.__threading:
        #     self.__threading = False
        #     self.runningThreadId.join()
        if self.__cap !=None:
            self.__cap.release()


    # def read(self,):
    #     """
    #     Read a frame from the image queue if available.
    #     """
    #     try:
    #         img = self.__image_queue.get_nowait()
    #     except queue.Empty:
    #         img = np.array([])
    #         pass
    #     return img

    def getShape(self):
        """
        Return the shape of the read images
        """
        if self.__cap != None:
            if self.__source_type == 'callable':
                success, img = self.__cap.read()[:2]
                if not success:
                    raise ValueError('[Error] Get image failed!')
                return img.shape
            elif self.__source_type == 'stream' or self.__source_type == 'video':
                return [int(self.__cap.get(4)), int(self.__cap.get(3)), 3]
        elif self.__files != None and len(self.__files)>0:
            img = cv2.imread(self._files[0])
            return img.shape
        else:
            raise ValueError("Unable to read from source")


    def __loadSource(self, source):
        """
        Determines the source type to load the correct reader
        Allowed sources are :
            callable source (have a read attribute)
            file source (the source is a file (opened with cv2.VideoCapture))
            stream source (the source is opened with cv2.VideoCapture. Example : rtsp source)

            Not properly implemented : directory source (the source is a direcctory that contains jpg images (alphanumeric order))
        """
        if hasattr(source, 'read'):
            self.__cap = source
            success, img = self.__cap.read()[:2]
            if not success:
                raise ValueError('[Error] Get image failed!')
            res = 'callable'
            self.__files = None
        elif os.path.isdir(source):
            self.__files = glob.glob(os.path.join(source, '*.jpg'))
            if len(self._files) == 0:
                raise ValueError('[Error] No image to load in folder')
            self.__files.sort()
            print('get %d images'%len(self.__files))
            res = 'dir'
            if self.__cap != None:
                self.cap.release()
                self.cap == None
        elif os.path.isfile(source):
            if self.__cap != None:
                self.cap.release()
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.__bufferSize)
            if cap.isOpened():
                self.__cap = cap
            else:
                raise ValueError('[Error] Open video [%s] failed!'%source)
            res = 'video'
            self.__files = None
        else:
            if self.__cap != None:
                self.__cap.release()
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.__bufferSize)
                self.__cap = cap
            else:
                raise ValueError('[Error] Open video [%s] failed!'%source)
            res = 'stream'
            self.__files = None
        return res

    def read(self):
        return self.frames[int(time.time()) % 3]


    def __readFromSource(self):
        """
        Read on the source
        Return : Bool, image (or None), timestamp (if available (callable source))
            True if the source is available and the image as been read, the image that has been read.
        """
        if self.__source_type == None:
            return False, None
        if self.__source_type == 'dir':
            if len(self.__files) > 0:
                return True, cv2.imread(self.__files.pop(0))
            else:
                return False, None
        else:
            return self.__cap.read()

if __name__ == '__main__':
    ip_camera1 = 'rtsp://192.168.0.151/media/video1'
    r = Reader(ip_camera1, 0)
    time.sleep(2)
    # r.__del__()
