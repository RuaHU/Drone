import json
from drone.pydrone import DJI, PARROT
import cv2
from Reader import Reader
import time
import random
def initDrone(droneController):
    if droneController == "PARROT":
        return PARROT()
    else:
        return DJI()

COLOR_LIST = ['rgb(186, 64, 147)', 'rgb(34, 124, 157)', 'rgb(23, 195, 178)', 'rgb(254, 203, 119)', 'rgb(254, 109, 115)','rgb(191, 215, 234)']

class ConfigLoader:
    def __init__(self, file):
        """
        Takes as input a json.file that describes the camera network
        """
        self.__config = file
        with open(self.__config) as json_file:
            data = json.load(json_file)
        self.__data = data

    def getConfig(self):
        """
        Return the config
        """
        return self.__data

    def getAllFixedCamera(self):
        """
        Return the config for "FixedCamera"
        """
        return self.__data["FixedCamera"][:]

    def getAllDroneCamera(self):
        """
        Return the config for "DroneCamera"
        """
        return self.__data["DroneCamera"][:]
    
    def getAllVideo(self,):
        '''
        Return the config for "Video"
        '''
        return self.__data["Video"][:]

    def getAsSource(self, cameraType):
        """
        Convert to useful form
        cameraType : FixedCamera | DroneCamera
        """
        res = {}
        for cam in self.__data[cameraType]:
            if cam["work"] == 0:continue
            if cameraType == "DroneCamera":
                droneModel = cam["droneController"]
                res[cam["id"]] = {"source": initDrone(droneModel), "altName": cam['altName'], "work":cam["work"]}
            elif cameraType == "FixedCamera":
                sourceDetected = False
                if "ip" in cam.keys():
                    source = cam["ip"]
                    sourceDetected = True
                if sourceDetected == False:
                    raise KeyError("Unable to detect source keys")
                alertZones = []
                if "alerts" in cam.keys():
                    alertZones = cam["alerts"]
                res[cam["id"]] = {"source": source, "altName": cam['altName'], "alerts": alertZones, "work":cam["work"]}
            elif cameraType == "Video":
                res[cam["id"]] = {"path" : cam['path'], 
                                  "altName":cam['altName'], 
                                  "timestamp":cam['timestamp'], 
                                  "work":cam["work"]}
            else:
                print(cam)
                print(cameraType)
                raise ValueError('unable to parse cameraType')
        return res

    def getStreams(self):
        """
                streams = [
                    {'id':0, 'drones':[2], 'latitude':40, "longitude":5},
                    {'id':1, 'drones':[2], 'latitude':40, "longitude":4}
                ]
        """
        res = []
        for fc in self.__data["FixedCamera"]:
            if fc["work"] == 0:continue
            droneSet = set()
            for alert in fc["alerts"]:
                for droneId in alert["droneId"]:
                    droneSet.add(droneId)
            res.append({'id':fc["id"],'drones':list(droneSet), "latitude":fc["latitude"], "longitude":fc["longitude"], "altName":fc["altName"]})
        for vc in self.__data['Video']:
            if vc["work"] == 0:continue
            res.append({'id':vc['id'],'drones':vc['droneId'], 'latitude':vc["latitude"], "longitude":vc["longitude"], "altName":vc['altName']})
            
        return res

    def colorMap(self):
        res = {}
        cpt = len(COLOR_LIST)
        for cam in self.getAllDroneCamera():
            if cpt == -1:
                res[cam["id"]] = "rgb(%d,%d,%d)" % (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            else:
                cpt -= 1
                res[cam["id"]] = COLOR_LIST[cpt]
        for cam in self.getAllFixedCamera():
            if cpt == -1:
                res[cam["id"]] = "rgb(%d,%d,%d)" % (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            else:
                cpt -= 1
                res[cam["id"]] = COLOR_LIST[cpt]
        return res

class zoneSelector:
    ix = -1
    iy = -1
    drawing = False

    def draw_rectangle_with_drag(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print("event")
            zoneSelector.drawing = True
            zoneSelector.ix = x
            zoneSelector.iy = y

        elif event == cv2.EVENT_MOUSEMOVE:
            #print("event2")
            if zoneSelector.drawing == True:
                param.imgCopy = param.img.copy()
                cv2.rectangle(param.imgCopy, pt1=(zoneSelector.ix,zoneSelector.iy), pt2=(x, y),color=(0,0,255),thickness=-1)

        elif event == cv2.EVENT_LBUTTONUP:
            #print("event3")
            zoneSelector.drawing = False
            param.imgCopy = param.img.copy()
            cv2.rectangle(param.imgCopy, pt1=(zoneSelector.ix,zoneSelector.iy), pt2=(x, y),color=(0,0,255),thickness=-1)
            print(zoneSelector.ix*param.originShape[1]/960, zoneSelector.iy*param.originShape[0]/540)
            print(x*param.originShape[1]/960,y*param.originShape[0]/540)
            print("===")
            print(min(x, zoneSelector.ix)*param.originShape[1]/960,min(y,zoneSelector.iy)*param.originShape[0]/540,(max(x, zoneSelector.ix)-min(x, zoneSelector.ix))*param.originShape[1]/960,(max(y, zoneSelector.iy)-min(y, zoneSelector.iy))*param.originShape[0]/540)

    def __init__(self, source, camid = 0):
        """
        Source to give to the reader
        HAVE TO DISPLAY : [x, y, w, h]
        """
        self.r = Reader(source, camid, True, type = "fixed")
        time.sleep(5)
        suc, img = self.r.read()
        self.img = img
        self.originShape = self.img.shape[:2]
        print(self.originShape)
        #cv2.rectangle(self.img, pt1=(1101, 398), pt2=(1101+303,398+190),color=(0,255,255),thickness=-1)
        self.img = cv2.resize(self.img, (960, 540))
        self.imgCopy = self.img.copy()
        cv2.namedWindow('Select two points to draw a rectangle')
        cv2.setMouseCallback('Select two points to draw a rectangle',zoneSelector.draw_rectangle_with_drag, self)
        while True:
            cv2.imshow('Select two points to draw a rectangle', self.imgCopy)
            if cv2.waitKey(10) == 27: #Escape
                break

if __name__ == '__main__':
    sampleConfigFile = "config.json"
    parser = ConfigLoader(sampleConfigFile)
    zs = zoneSelector("rtsp://192.168.0.104/media/video1")
    cv2.destroyAllWindows()

"""
"""
