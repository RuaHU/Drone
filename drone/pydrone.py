#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:33:51 2021

@author: hu
"""
from .TELLO import Tello
import cv2
import numpy as np
from .pyparrot.Anafi import Anafi
import time
import _thread
import queue

class Drone:
    def __init__(self,
                 drone_ip : str = None):

        self._connected = False
        self._ip = drone_ip
        self.move_direction = ['forward',
                               'back',
                               'right',
                               'left',
                               'down',
                               'up']

    #virtual function list
    def connect(self,):return self._connect()
    def takeoff(self,):self._takeoff()
    def land(self,):self._land()
    def move(self, *args, **kwargs):self._move(*args, **kwargs)
    def rotate(self, *args, **kwargs):self._rotation(*args, **kwargs)
    def move_camera(self, *args, **kwargs):self._move_camera(*args, **kwargs)
    def battery(self,):return self._battery()
    def disconnect(self,):self._disconnect()
    def isOpened(self,):self._isOpened()
    def idle(self,):self._idle()
    def read(self, timestamp = True):
        '''opencv-videocapture-like interface'''
        if timestamp:
            img, t = self._read(True)
            if img is None:
                return False, np.array([]), t
            if img.size == 0:
                return False, np.array([]), t

            return True, img, t
        else:
            img = self._read(False)
            if img is None:return False, np.array([])
            if img.size == 0:return False, np.array([])
            return True, img
    #demonstration && testing && basic movement
    def demo_video(self,):self._video_loop()
    def demo_video_thread(self,):
        _thread.start_new_thread(self._video_loop, ('',))
    def demo_mv1(self,):
        if not self._connected:
            print('connect first!')
            return
        self._takeoff()
        self._battery()
        self._land()
    def demo_mv2(self,):
        if not self._connected:
            print('connect first!')
            return
        self._takeoff()
        for direction in self.move_direction:
            self._move(direction, 0.5)
        self._land()
    def demo_mv3(self,):
        if not self._connected:
            print('connect first!')
            return
        self._takeoff()
        self._rotation(180)
        self._rotation(-180)
        self._land()

    def demo_rotation(self,):
        self._takeoff()
        angle = 180
        _angle = angle
        while _angle > 0:
            _angle -= 30
            self._rotation(30)
        _angle = angle
        while _angle > 0:
            _angle -= 30
            self._rotation(-30)
        self._land()

class DJI(Drone):
    def __init__(self,):
        super(DJI, self).__init__(drone_ip = '192.168.10.1')
        self._local_ip = ''
        self._local_port = 8889
        self._port = 8889

    def _idle(self):
        self.drone.idle()

    def _isOpened(self):
        if self._connected:
            return True

    def _connect(self,):
        self.drone = Tello(self._ip)
        self.drone.connect()
        #receive video stream
        self.drone.streamon()
        self._connected = True
        #self.drone.set_speed(20)
        return True

    def _takeoff(self,):
        self.drone.takeoff()
    def _land(self,):
        self.drone.land()
    def _move(self, direction, distance):
        #distance m
        self.drone.move(direction, int(distance * 100))
    def _rotation(self, angle):
        if angle >= 0:
            self.drone.rotate_clockwise(angle)
        else:
            self.drone.rotate_counter_clockwise(abs(angle))

    def _soft_rotation(self, speed):
        '''speed rotation speed [-100, 100]'''
        self.drone.send_rc_control(0, 0, 0, speed)
    def _move_camera(self,):
        '''tello does has a fixed camera '''
        print('change camera viewport failed!')
        return

    def _read(self, timestamp = True):
        '''read video frame'''
        return self.drone.read(timestamp = timestamp)

    def _video_loop(self, name = None):
        if not self._connected:
            print('connect first!')
            return
        while True:
            frame = self.drone.read()
            cv2.imshow('tello', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _battery(self,):        return self.drone.get_battery()

    def _print_battery(self,):  print(self.drone.get_battery())

    def _height(self,):         return self.drone.get_height()

    def _print_height(self,):   print(self.drone.get_height())

    def _speed(self,):          return self.drone.get_speed()

    def _print_speed(self,):    print(self.drone.get_speed)

    def _disconnect(self,):
        del self.drone
        self._connected = False

class PARROT(Drone):
    def __init__(self,):
        super(PARROT, self).__init__(drone_ip = '192.168.42.1')
        self._setting()
        self.source ='rtsp://%s/live'%self._ip
        self._image_queue = queue.Queue(10)

    def _idle(self):
        pass

    def _isOpened(self):
        return self.cap.isOpened()

    def _cap(self,):
        if hasattr(self, 'cap'):self.cap.release()
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            self.cap = cap
            self.shape = [cap.get(4), cap.get(3), 3]
        else:
            raise ValueError('[Error] Open video [%s] failed!'%self.source)

    def _read(self, timestamp = True):
        if timestamp:
            return self._image_queue.get()
        else:
            return self._image_queue.get()[0]

    def _video_thread(self,):
        self._cap()
        while True:
            if not self._connected:
                time.sleep(0.02)
            success, img = self.cap.read()
            timestamp = time.time()
            if not success:
                self._cap()
            else:
                frame = [img, timestamp]
                try:
                    self._image_queue.put_nowait(frame)
                except queue.Full:
                    #fresh the queue
                    self._image_queue.get_nowait()
                    self._image_queue.put_nowait(frame)
                    pass

    def _setting(self,):
        '''anafi'''
        self.speed = 2 #m/s
        self.rotation_speed = 180 #degree/s
        self.drone = Anafi('Anafi', self._ip)
        #self.drone.set_max_vertical_speed(self.speed)
        #self.drone.set_max_rotation_speed(self.rotation_speed)
    #commands

    def _video_loop(self, name = None):
        if not self._connected:
            print('connect first!')
            return
        self._cap()
        while True:
            success, frame = self.cap.read()
            cv2.imshow('tello', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):break
            if not success:break

    def _connect(self,):
        self.drone.connect(10)
        self._connected = True
        self.drone.set_gimbal_target(0, 0, 0, 0)
        _thread.start_new_thread(self._video_thread, ())
        return True
    def _takeoff(self,):
        self.drone.safe_takeoff(5)
    def _land(self,):
        self.drone.safe_land(5)
    def _move(self, direction, distance):
        assert distance >= 0
        assert direction in self.move_direction
        command = [0, 0, 0, 0]
        index = self.move_direction.index(direction)
        index, to = index // 2, index % 2
        command[index] = np.sign(0.5 - to) * distance
        print(direction, command)
        self.drone.move_relative(*command)
    def _rotation(self, direction):
        '''degree'''
        #degree to radians
        direction = direction * 3.1415 / 180
        self.drone.move_relative(0, 0, 0, direction)
        #direction / self.rotation_speed
        #self.drone.fly_direct(0, 0, 20, 0, 0.1)
    def _move_camera(self, direction):
        self.drone.set_gimbal_target(0, 0, direction, 0)
    def _battery(self,):return 0
    def _disconnect(self,):
        self.drone.disconnect()
        self._connected = False


if __name__ == '__main__':
    anafi = PARROT()
    anafi.connect()
    anafi._video_loop()
    anafi.demo_mv1()
    anafi.demo_mv2()
    anafi.demo_mv3()
    anafi.demo_rotation()
    anafi.disconnect()

#if __name__ == '__main__':
#    tello = DJI()
#    tello.connect()

    '''
    print('drone takeoff {}'.format(time.time()))
    tello.takeoff()
    print('drone takeoff {}'.format(time.time()))
    tello._print_battery()
    tello._print_height()
    tello.move('up', 0.6)
    print('drone takeoff {}'.format(time.time()))
    tello._print_height()
    tello.move('down', 0.5)
    tello.move('forward', 0.5)
    tello.move('back', 0.5)
    tello.move('left', 0.5)
    tello.move('right', 0.5)
    tello.rotate(10)
    tello.rotate(-180)
    tello._print_height()
    tello.land()
    tello.demo_rotation()
    '''
    #tello._video_loop()
    #tello.demo_video_thread()
    #tello.demo_mv1()
    #tello.demo_mv2()
    #tello.demo_mv3()
    #tello.demo_rotation()
#    tello.disconnect()
