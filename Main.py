#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Reader import Reader, Videos, VideoSource
from ConfigLoader import ConfigLoader
import queue
import threading
import cv2
import numpy as np


from Detector import Detector
from Tracker1 import Tracker
from Package import Package, PackageList
from backend import get_drone, getAll_drones
import time
import datetime
"""
http://www.laurentluce.com/posts/python-threads-synchronization-locks-rlocks-semaphores-conditions-events-and-queues/
https://github.com/Akbonline/Autonomous-Drone-based-on-Facial-Recognition-and-Tracking
Il faudrait mettre de l autenthicaion pour droneapi

https://stackoverflow.com/questions/64178746/flexbox-css-layout-with-header-footer-partly-scrollable-sidebar

processedToStreamable : Trouver un moyen de kill le processus si le client n est pas co parce que la j'ai duplication de ces services ?
lorsque le drone est connectée, perte d 'acces aux apis googleapis => lenteur pdt la recherche' --> desactivation des googles fonts + jquery en local
"""
TRACK = True
REID = False
COUNT = True
DEPLOY_LOCAL = False
GLOBAL_CONNECT_FLASK = True
if GLOBAL_CONNECT_FLASK:
    from flask import Flask, render_template, Response, request, jsonify
    import json

logfile = open('debug.log', 'w')


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
                    #keeps 20 frames at most to avoid memory overflow
                    self.__imagePool[_producer.id] = self.__imagePool[_producer.id][-20:]
            time.sleep(0.02)


class detectionAndTrackingPool(threading.Thread):
    '''
    detect images once an image come into the detection queue
    '''
    def __init__(self,
                 _detector : Detector,
                 _imageProducer_dict : dict,
                 ):
        threading.Thread.__init__(self)
        self.__imageProducer_dict = _imageProducer_dict
        self.detector = _detector
        self.__detection_time_elispe = -1
        
    def test(self,):
        image = cv2.imread('demo/1.jpg')
        detection = self.detector.det(image)
        print(detection[0].shape, detection[1].shape, detection[2].shape)
        #image = Detector.draw(image, detection[1][0])
        #cv2.imshow('detection', image)
        #cv2.waitKey(0)
    
    def run(self,):
        while True:
            validPkgs = []
            for _camId in self.__imageProducer_dict:
                _producer = self.__imageProducer_dict[_camId]
                pkg = _producer.packageList.getValidDetectionPackage()
                if pkg is not None:
                    validPkgs.append(pkg)
            
            if len(validPkgs) > 0:
                '----------------------detection--------------------------'
                #print('validPkgs:', len(validPkgs))
                begin = time.time()
                self.detector.detPkgs(validPkgs)
                #update detection time consuming
                self.__detection_time_elispe = time.time() - begin if self.__detection_time_elispe == -1 \
                    else (self.__detection_time_elispe + time.time() - begin)/2
                print('detection time elispe: {}'.format(self.__detection_time_elispe))
                
                '----------------------tracking--------------------------'
                for pkg in validPkgs:
                    _producer = self.__imageProducer_dict[pkg.id]
                    _producer.tracker.solver(pkg)
                    _producer.packageList.fillUpPackages(_producer.tracker)
                    
            else:
                time.sleep(0.02)

def setAllCameras(camConfig):
    fixedData = camConfig.getAsSource('FixedCamera')
    droneData = camConfig.getAsSource('DroneCamera')
    videoData = camConfig.getAsSource('Video')
    _index2id_map = []
    _cameraReaders = []
    for camId in fixedData :
        if fixedData[camId]["work"] == 0:continue
        _cameraReaders.append(Reader(fixedData[camId]["source"], camId, True, fixedData[camId]["altName"], _type = "fixed"))
        _index2id_map.append(camId)
    for camId in droneData :
        if droneData[camId]["work"] == 0:continue
        _cameraReaders.append(Reader(droneData[camId]["source"], camId, False, droneData[camId]["altName"], _type="drone"))
        _index2id_map.append(camId)
    
    if len(videoData) > 0:
        videoDict = {}
        for camId in videoData:
            if videoData[camId]["work"] == 0:continue
            videoDict[camId] = [videoData[camId]['path'], videoData[camId]['timestamp']]
            _index2id_map.append(camId)
        videos = Videos(videoDict)
        #start video reading
        videos.start()
        for camId in videoData:
            if videoData[camId]["work"] == 0:continue
            _cameraReaders.append(Reader(VideoSource(videos, camId), camId, True, videoData[camId]["altName"], _type="video"))
    _delimitation = len(_cameraReaders)
    return _cameraReaders , _delimitation, _index2id_map

def setAlertZonesDict(camConfig):
    """
    Only for fixedCam
    """
    fixedData = camConfig.getAsSource('FixedCamera')
    res = {}
    for camId in fixedData :
        res[camId] = fixedData[camId]["alerts"]
    return res

def remapWithIndex(listOfdict, index2id, keyList = ['id','drones']):
    """
    """
    for key in keyList:
        for _dict in listOfdict:
            if key in _dict:
                if isinstance(_dict[key],int):
                    _dict[key] = index2id.index(_dict[key])
                else:
                    for i in range( len(_dict[key])):
                        _dict[key][i] = index2id.index(_dict[key][i])
    return listOfdict



if __name__ == '__main__':
    configFile = 'config.json'
    camConfig = ConfigLoader(configFile)
    cameraReaders, delimitation, index2id_map = setAllCameras(camConfig)
    
    _detector = Detector('yolov4-tiny')
    _cameraImageProducerDict = {_reader.getId():cameraImageProducer(_reader) for _reader in cameraReaders}
    _detectionAndTrackingPool = detectionAndTrackingPool(_detector, _cameraImageProducerDict)
    #do a test for object detection
    _detectionAndTrackingPool.test()
    
    _flaskWebImageConsumer = flaskWebImageConsumer(_cameraImageProducerDict)
    
    #start threadings
    for _camId in _cameraImageProducerDict:
        _producer = _cameraImageProducerDict[_camId]
        try:
            _producer.start()
        except:
            raise ValueError('unable to create thread')
        
    
    _detectionAndTrackingPool.start()
    _flaskWebImageConsumer.start()
        

    if GLOBAL_CONNECT_FLASK:

        app = Flask(__name__)
        
        streams = camConfig.getStreams()#remapWithIndex(camConfig.getStreams(), index2id_map)
        colors = [camConfig.colorMap()]#remapWithIndex([camConfig.colorMap()], index2id_map)[0]

        def processedToStreamable(camid):
            """
            return the right image
            """
            print("LE PROCESSUS DE STREAMING A ETE LANCE {}".format(camid))
            while True:
                item = _flaskWebImageConsumer.getFrame(camid)
                if item == False:continue
                frame, timestamp = item
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                

        GLOBAL_COUNT_DATA = {}
        for cam in Reader.knownCameras.keys():
            GLOBAL_COUNT_DATA[cam] = {"value":0}
            
        def getCount():
            pastTime = time.time()
            while True:
                for cam in Reader.knownCameras.keys():
                    data = Reader.knownCameras[cam].getData('detected') # [time, len(outBoxes)]

                    if data != None:
                        if isinstance(data[0], datetime.datetime):
                            GLOBAL_COUNT_DATA[cam] = {"time": data[0].strftime("%H:%M:%S"), "value": max(data[1], GLOBAL_COUNT_DATA[cam]["value"])}
                        else:
                            GLOBAL_COUNT_DATA[cam] = {"time": time.strftime("%H:%M:%S", data[0]), "value": max(data[1], GLOBAL_COUNT_DATA[cam]["value"])}
                    else:
                        GLOBAL_COUNT_DATA[cam] = {"value":0}


                curTime = time.time()
                if (curTime - pastTime) > 10:
                    json_data = json.dumps(GLOBAL_COUNT_DATA)
                    pastTime = curTime
                    for key in GLOBAL_COUNT_DATA:
                        GLOBAL_COUNT_DATA[key] = {"value":0}
                    yield f"data:{json_data}\n\n"


        @app.route('/index')
        @app.route('/')
        def index():
            return render_template('index.html', streams=streams)

        @app.route('/video_feed/<int:camid>')
        def video_feed(camid=0):
            return Response(processedToStreamable(camid),mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route("/chart-data")
        def chart_data():
            return Response(getCount(), mimetype="text/event-stream")

        @app.route('/camera_monitorOLD/')
        def camera_monitorOLD():
            if request.method == 'GET':
                for i, stream in enumerate(streams):
                    if stream['id'] ==  request.args.get('camid', default=0, type=int):
                        return render_template('cameraAppView.html', camid=request.args.get('camid', default=0, type=int), drones=stream["drones"])
                    else:
                        return render_template('index.html', streams=streams)
            if request.method == 'POST':
                pass

        @app.route('/camera_monitor/')
        def camera_monitor():
            if request.method == 'GET':
                for i, stream in enumerate(streams):
                    if stream['id'] ==  request.args.get('camid', default=0, type=int):
                        #mainViewId = request.args.get('mainid', default=0, type=int)
                        return render_template('cameraAppView.html', mainid=request.args.get('mainid', default=0, type=int), drones=stream["drones"]+[stream['id']], camid =stream['id'])
            return render_template('index.html', streams=streams)

        @app.route('/camera_monitor_demo/')
        def camera_monitor_demo():
            if request.method == 'GET':
                for i, stream in enumerate(streams):
                    if stream['id'] ==  request.args.get('camid', default=0, type=int):
                        #mainViewId = request.args.get('mainid', default=0, type=int)
                        return render_template('demoView.html', mainid=request.args.get('mainid', default=0, type=int), drones=stream["drones"]+[s['id'] for s in streams], camid =stream['id'], allData = streams, colors=colors)
            return render_template('index.html', streams=streams)


        @app.route('/droneapi/command/', methods=['POST'])
        def command():
            cmd = request.form.get('command')
            droneId = int(request.form.get('droneId'))
            drone = get_drone(droneId)
            """
            Requires the id from the json and not the one used into the main
            """
            print(drone)
            print(cmd)
            if drone != None:
                #print(drone.drone.get_height())
                if cmd == 'takeOff':
                    drone.takeoff()
                    print("taking off")
                if cmd == 'land':
                    drone.land()
                    print("landing")
                    """
                    Penser à reset l etat dans ls boucles principales
                    """
                if cmd == 'up':
                    drone.move('up', 0.5)
                if cmd == 'down':
                    drone.move('down',0.5)
                if cmd == 'right':
                    print("command right")
                if cmd == 'left':
                    print("command left")
                if cmd == 'ccw':
                    drone.rotate(-10)
                if cmd == 'cw':
                    drone.rotate(10)
                if cmd == 'forward':
                    print("command forward")
                if cmd == 'back':
                    print("command back")
            if cmd == 'landall':
                for drone in getAll_drones():
                    drone.land()
            if cmd == "emergency":
                # see tello api but Parrot ?
                pass
            return jsonify(status='success'), 200

        @app.route('/trackingapi/params/',  methods=['POST'])
        def tracker_options():
            trackingOptionCmd = request.form.get('command')
            print(trackingOptionCmd)
            return jsonify(status='success'), 200

        @app.route('/fixedcams')
        def fixed_cam():
            pass

        @app.route('/allcams')
        def all_cams():
            pass

        if DEPLOY_LOCAL:
            app.run('0.0.0.0', 8080, debug=True, use_reloader=False)
        else:
            app.run(debug=True, use_reloader=False)
