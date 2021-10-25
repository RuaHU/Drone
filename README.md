### Cross Camera Tracking
### Global Program Architecture
![Global Program Architecture](https://github.com/RuaHU/Drone/blob/main/Documents/ProgramArchitecture.jpg?raw=true)

### Demo
A fixed camera for indoor person tracking

A mobile camera (drone) for outdoor person tracking

When a person go to the alert region (manually specified rectangle) will trigger the drone to continue track the person and change it's viewport to center the person.

![Demo](https://github.com/RuaHU/Drone/blob/main/Documents/demo.jpg?raw=true)
[full video](https://drive.google.com/file/d/1yBFwXZypTnXhuR1ffHYsBypDvkpxU8zr/view?usp=sharing)

**Table of Contents**

[TOCM]

[TOC]
# Development Documentation
## Installation
### Environment
- conda==4.10.1
- Flask==2.0.1
- h264decoder==0.0.0
- h5py==2.10.0
- Keras==2.2.4
- lap==0.4.0
- mish-cuda==0.0.3
- numpy==1.19.1
- opencv-contrib-python==4.1.1.26
- opencv-python==4.5.1.48
- tensorflow==1.14.0
- torch==1.2.0
- torchvision==0.4.0a0+6b959ee
- wxPython==4.0.4
### Usage
```python
python Main.py
```
Follow the prompts and enter 127.0.0.1:5000 in the browser to open the user interface and browse the surveillance videos.

## Modules
This program including 4 modules.
- Detector: Person detection and re-identification feature extraction
- Tracker: Person tracking
- Reader: Data interface [stream, videos].
- Package: Encapsulate data for Detector and Tracker
### Detector
The detector is defined in [Detector.py](https://github.com/RuaHU/Drone/blob/ee773e3e9124d8ef9b38bdc627b33650206240c4/Detector.py) for person detection and person re-identification feature extraction.
#### Usage
A quick check of the Detector module.
```python
python Detector.py
```
The detailed usage of the detector
```python
detector = Detector('yolov4-tiny')
image = cv2.imread('demo/1.jpg')
detection = detector.det(image)
```
This example using yolov4-tiny as detector to get detection of the input image demo/1.jpg.
The detector could be  ['yolov3', 'yolov4', 'yolov3-tiny', 'yolov4-tiny', 'dla_34', 'mrcnn'].

#### Custom Detector
##### Custom Parameters for Existing Detectors
The parameters of detectors are defined in [Detector.Config](https://github.com/RuaHU/Drone/blob/ee773e3e9124d8ef9b38bdc627b33650206240c4/Detector.py)

```python
if self.M == 'yolov4-tiny':
            self.IMAGE_MAX_DIM = 416
            self.IMAGE_SIZE = [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM]
            self.DETECTION_MIN_CONFIDENCE = 0.2
            self.DETECTION_NMS_THRESHOLD = 0.3
            self.mgn = False
```
The parameters like the input image size and the threshold for NMS can be modified here.
##### Introduce a Custom Detector
Any other detector can be used to replace the existing detector. The input and outputs of the detector are defined as follow.
```python
'''
        input: image, [B, W, H, 3]
        output: 
            1- re-identification feature vectors. [B, N, D]
            2- detection bounding boxes. [B, N, 4]
            3- detection scores. [B, N, 1]
        '''
```
### Tracker
The tracker is defined in [Tracker.py](https://github.com/RuaHU/Drone/blob/ee773e3e9124d8ef9b38bdc627b33650206240c4/Tracker1.py) for person tracking.
Use Kalman filter ([Kalman](https://github.com/RuaHU/Drone/blob/ee773e3e9124d8ef9b38bdc627b33650206240c4/Tracker1.py)) to smooth and predict trajectory. Unlike other Kalman filters based on continuous frames and fixed time intervals, the time interval of the state transition matrix of the Kalman filter we use depends on the time interval between frames, so that the target state at any moment can be predicted and Filtering.
```python
def predict(self, _time_increment = 1):
        '''_time_increment : interval between two frames'''
        assert self.predicted == False
        ndim = 4
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):self._motion_mat[i, ndim + i] = _time_increment
        self.prediction = self.predict_internal(self.mean, self._motion_mat)
        self.predicted = True
        return self.prediction
```
#### Usage
[Tracker.py](https://github.com/RuaHU/Drone/blob/ee773e3e9124d8ef9b38bdc627b33650206240c4/Tracker1.py)
```python
def solver(self,
               _pkg : Package)-> None:
```
is the user interface of the tracker. Input a package that has been detected by the detector, and the tracker will track the detections. And fill in the tracking results into the package. (Notice: the tracking results filled in the input parameter _pkg)

#### Tracking Process
![track state machine](https://github.com/RuaHU/Drone/blob/c05e8731f259b6b776b2e4b28b213f3c69567324/Documents/statemachineandassociationpipeline.jpg?raw=true)
Above is the states of tracks.

The key to solving the tracking problem is to find the matches between the track and the detections.

Use the following steps to find the matches:
- 1 Find matches based on the similarities of re-identification features between tracks and the detections.
```python
'---------------------------first association---------------------'
        if len(track_list) == 0 or len(frame) == 0:
            similarities = np.zeros((len(track_list), len(frame)), dtype = np.float32)
        else:
            similarities = probes @ gallery.T
        distance_matrix = np.zeros((len(track_list), len(frame)), dtype = 'float32')
        for i, t in enumerate(track_list):
            distance_matrix[i, :] = t.kf.gating_distance(frame[:, :4].copy())
        cost_matrix = 1 - similarities
        cost_matrix[distance_matrix > chi2inv95[9]] = np.inf
        cost_matrix[cost_matrix > 0.5] = np.inf
        cost_matrix = _lambda * cost_matrix + (1 - _lambda) * distance_matrix
        matches, ut, ud = self.match(cost_matrix, jde_thresh)
        self._update(track_list, frame, matches, _pkg)
```
- 2 Find matches based on the IOU between the predictions of the tracks and the detection
```python
        iou_matrix = np.zeros((len(ut), len(ud)), dtype = 'float32')
        for i, t in enumerate(tracks): iou_matrix[i, :] = t.kf.gating_iou(frame[:, :4].copy())
        matches, ut, ud = self.match(1-iou_matrix, iou_thresh)
        self._update(tracks, frame, matches, _pkg)
```

#### Debug Tracker
Target tracking needs to process multiple frames of images, and the designed tracker is difficult to debug in a complete system. We use [debug_tracker.py](https://github.com/RuaHU/Drone/blob/c05e8731f259b6b776b2e4b28b213f3c69567324/debug_tracker.py) to debug the tracker.

In an actual system, the detector is always much slow than then realtime stream, it is impossible to detect each frame. The tracker will not track consecutive frames, the tracked frames are randomly spaced according to the detection speed. Therefore, the performance of the tracker is still unstable. The specific manifestations are loss tracking, ID switch and other cases.
We provide a simple debugger to improve the design of the tracker. Unfortunately, there is a big gap between the debugger and the actual performance. Need to be further improved.

### Reader
Reader is defined in [Reader.py](https://github.com/RuaHU/Drone/blob/c05e8731f259b6b776b2e4b28b213f3c69567324/Reader.py). The purpose of designing Reader is to manage the interface of video stream reading in a unified way, and to provide the same video stream as the real-time runtime during program debugging. As multiple cameras are involved, Reader needs to synchronize each video stream.

The parameters of reader are:
```python
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
```
where _source is the source of video stream. It could be a rtsp video stream address or a wrapped object that has .read() callable attribution.

where _bRecording is set True to record the videos, this is specifically for system debugging.
the recorded video will be saved in records/.../output.avi and records/.../timestamp.txt.
####Example Read from Stream
```python
reader = Reader("rtsp://192.168.0.151/media/video1", 1)
image, timestamp = reader.read()
```

####Example Read from Videos
```python
video_dict = {1:[path/to/video/.avi, path/to/timestamp/.txt],
                        2:[path/to/video/.avi, path/to/timestamp/.txt]}
videos = Videos(video_dict)
reader1 = Reader(VideoSource(videos, 1), 1)
reader2 = Reader(VideoSource(videos, 2), 2)
image, timestamp = reader1.read()
image, timestamp = reader2.read()
```
The above code realizes the synchronization access of videos according to timestamp files.

### Package
Package is defined in [Package.py](https://github.com/RuaHU/Drone/blob/c05e8731f259b6b776b2e4b28b213f3c69567324/Package.py) to wrap data and serves as input for detector and tracker.

```python
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
```
The package connects Reader, Detector, and Tracker. The main idea of this system design is to deal with these packages.

Every time an image is read from Reader, a package is initialized. If the detector and tracker in the system are idle, this data package will be processed by the detector and tracker, otherwise the detection and tracking filtering will be abandoned.

We define a PackageList for each video source to manage data packages. The PackageList is defined in [Package.py](https://github.com/RuaHU/Drone/blob/c05e8731f259b6b776b2e4b28b213f3c69567324/Package.py).

A core function of PackageList is to fill data for the detection and tracking of non-contiguous frames, and fill the data of untracked packages by linear interpolation or Kalman filter prediction. In this way, we can obtain visually continuous trajectories. Here is the code:
```python
def fillUpPackages(self, tracker):
        self._tide_packageList()
        index = -1
        for i, pkg in enumerate(self.packageList[1:]):
            if pkg.getStates()[0]:
                index = i
                break
        if index <= -1:return
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
```

### Main
The overall framework of the program [Main.py](https://github.com/RuaHU/Drone/blob/c05e8731f259b6b776b2e4b28b213f3c69567324/Main.py) is as follows:
- 1 A thread get image from reader and encapsulate data by Package, and stack the package in PackageList.
```python
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
```
- 2 Get data from packageList and detect and track.
```python
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
```
- 3 Get packages from PackkageList and draw tracking results. The draw image will then be used by Flask.
```python
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
```
### End
