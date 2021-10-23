from Reader import Reader, Videos, VideoSource

import numpy as np

from Detector import Detector
from Tracker1 import Tracker
from Package import Package, PackageList

from ConfigLoader import ConfigLoader

import threading
import time

class cameraImageProducer(threading.Thread):
    """
    Classe allant produire l'image finale et notifier à chaque fois que l'image est prête
    """
    def __init__(self, reader,
                 ):
        threading.Thread.__init__(self)
        self.reader = reader
        self.tracker : Tracker = Tracker()
        self.packageList = PackageList()
        self.id = self.reader.getId()
            
    def run(self,):
        while True:
            success, img, timestamp = self.reader.read()
            if success:
                self.packageList.stackPackage(Package(img, timestamp, self.id))
                while len(self.packageList.packageList) > 20:
                    time.sleep(0.02)
            else:
                time.sleep(0.02)

class DebugTracker:
    def __init__(self,):
        camConfig = ConfigLoader('config.json')
        self._detector = Detector('yolov4-tiny')
        self.packageList = PackageList()
        self.readers = self.setAllCameras(camConfig)
        self._producer = cameraImageProducer(self.readers[0])
        self._producer.start()
        
    def setAllCameras(self, camConfig):
        videoData = camConfig.getAsSource('Video')
        videoDict = {}
        _cameraReaders = []
        for camId in videoData:
            if videoData[camId]["work"] == 0:continue
            videoDict[camId] = [videoData[camId]['path'], videoData[camId]['timestamp']]
        videos = Videos(videoDict)
        #start video reading
        videos.start()
        for camId in videoData:
            if videoData[camId]["work"] == 0:continue
                
            _cameraReaders.append(Reader(VideoSource(videos, camId), camId, True, videoData[camId]["altName"], _type="video"))
        return _cameraReaders
    
    def tracking(self,):
        while True:
            pkg = self._producer.packageList.getValidDetectionPackage()
            if pkg is not None:
                self._detector.detPkg(pkg)
                self._producer.tracker.solver(pkg)
                self._producer.packageList.fillUpPackages(self._producer.tracker)


class flaskWebImageConsumer(threading.Thread):
    def __init__(self,
                 _imageProducer_dict : dict,
                 ):
        threading.Thread.__init__(self)
        self.__imageProducer_dict = _imageProducer_dict
        self.__imagePool = {_camId:[] for _camId in self.__imageProducer_dict}
    
    def getFrame(self, camid):
        if len(self.__imagePool[camid]) == 0:return False
        return self.__imagePool[camid].pop(0)
        
    def run(self,):
        while True:
            for _camId in self.__imageProducer_dict:
                _producer = self.__imageProducer_dict[_camId]
                rets = _producer.packageList.play()
                if rets is not False:
                    #rets: image, timestamp
                    self.__imagePool[_producer.id].append(rets)
            time.sleep(0.02)

if __name__ == '__main__':
    debug = DebugTracker()
    debug.tracking()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            