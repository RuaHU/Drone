#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import lap
import numpy as np
import scipy.linalg
import itertools

from typing import List
from Package import Package

def xywh2ccwh(boxes):
    """
    From x,y, width height to center, center,width, height
    Parameters :
        boxes : nx4 np.ndarray
    """
    boxes = boxes.copy()
    boxes[:, :2] += (boxes[:, 2:4] * 0.5)
    return boxes


# vim: expandtab:ts=4:sw=4

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class Kalman(object):
    def __init__(self,):
        ndim = 4
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 200
        self._std_weight_velocity = 1. / 160
        self.initialed = False
        self.predicted = False

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        self.mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        self.covariance = np.diag(np.square(std))

    def getBoxState(self,):
        mean = self.mean.copy()[:4]
        mean[2:3] *= mean[3:]
        return mean

    def predict(self, _time_increment = 1):
        '''dt : interval between two frames'''
        assert self.predicted == False
        ndim = 4
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):self._motion_mat[i, ndim + i] = _time_increment
        
        self.prediction = self.predict_internal(self.mean, self._motion_mat)
        
        self.predicted = True
        return self.prediction
    
    def predict_internal(self, _mean, _motion_mat):
        prediction = np.dot(_mean, _motion_mat.T)[:4]
        prediction[2:3] *= prediction[3:4]
        return prediction
    
    def predict_external(self, _time_increment):
        ndim = 4
        _motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):_motion_mat[i, ndim + i] = _time_increment
        return self.predict_internal(self.mean, _motion_mat)
    
    def project(self, update = False):
        assert self.predicted == True
        std_pos = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-2 * np.ones_like(self.mean[3]),
            self._std_weight_position * self.mean[3]]
        std_vel = [
            self._std_weight_velocity * self.mean[3],
            self._std_weight_velocity * self.mean[3],
            1e-5 * np.ones_like(self.mean[3]),
            self._std_weight_velocity * self.mean[3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.diag(sqr)

        mean = np.dot(self.mean, self._motion_mat.T)

        left = np.dot(self._motion_mat, self.covariance)
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        if update:self.mean, self.covariance = mean, covariance

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov


    @staticmethod
    def toxyxy(m):
        m[..., :2] -= 0.5 * m[..., 2:]
        m[..., 2:] += m[..., :2]
        return m
    
    @staticmethod
    def iou(b1, b2):
        boxes1, boxes2 = b1.copy(), b2.copy()
        boxes1 = Kalman.toxyxy(boxes1)
        boxes2 = Kalman.toxyxy(boxes2)
        yx_min = np.maximum(boxes1[..., :2], boxes2[..., :2])
        yx_max = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        it = np.clip((yx_max - yx_min), 0, 1e10)
        ai = it[..., 0] * it[..., 1]
        ab = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        ar = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        return ai / (ab + ar - ai)

    def gating_iou(self, m, res=0):
        measurements = m.copy()
        prediction = self.prediction.copy()
        prediction[:2] += res
        return self.iou(prediction, measurements)

    def gating_distance(self, measurements, 
                        only_position=False, 
                        metric='maha'):
        measurements = measurements.copy()
        measurements[:, 2] /= measurements[:, 3]
        mean, covariance = self.project()
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        d = (measurements - mean)
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

    def update(self, node, res = 0, update_iou = True, factor = 1):
        measurement = node._get_detection().copy()
        measurement[2] /= measurement[3]
        if self.initialed == False:
            self.initiate(measurement)
            self.initialed = True
            self.predicted = False
            node._set_filtering(self.getBoxState())
            return

        self.mean[:2]+=res
        projected_mean, projected_cov = self.project(True)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(self.covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = (measurement - projected_mean) * factor

        self.mean = self.mean + np.dot(innovation, kalman_gain.T)
        self.covariance = self.covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        self.predicted = False
        node._set_filtering(self.getBoxState())


class Node():
    '''A node is a point in a track, it recodes the basic information'''
    
    def __init__(self, 
                 frame_id,
                 detection,
                 feature,
                 score,
                 timestamp,
                 ):
        self.__frame_id = frame_id      #frame id of current frame
        self.__timestamp = timestamp    #timestamp of current frame
        self.__detection = detection    #detection bounding box [cx, cy, w, h]
        self.__feature = feature        #re-identification feature of current bounding box
        self.__score = score            #detection score of current box
        self.__filtering = None         #tracking/filtering bounding box, [cx, cy, w, h]
        
    def _set_filtering(self, _filtering):
        self.__filtering = _filtering
    
    def _set_detection(self, _detection):
        self.__detection = _detection
    
    def _get_filtering(self,):
        assert self.__filtering is not None
        return self.__filtering
    
    def _get_detection(self,):
        return self.__detection
    
    def _get_feature(self,):
        return self.__feature
    
    def _get_score(self,):
        return self.__score
    
    def _get_timestamp(self,):
        return self.__timestamp
    
    def _get_frame_id(self,):
        return self.__frame_id

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Invalid = 3
    Removed = 4

class Track:
    count = 0
    def __init__(self, 
                 node : Node,
                 ):
        self.kf = Kalman()
        self.id = -1
        self.kf.update(node)
        self.T = [node]
        self.frame_id = node._get_frame_id()
        self.features = [[0, 0, node._get_feature()],
                         [0, 0, node._get_feature()],
                         [0, 0, node._get_feature()]]
        self.state = TrackState.New
        self.is_activated = False
        
        self.max_time_lost = 30
        self.max_score = 0.
        self.max_area= 0.
        self.profile = []
    
    def getFilteredResults(self,):
        return self.kf.getBoxState()
        
    def predict(self, timestamp_increment):
        self.prediction = self.kf.predict(timestamp_increment)
        return self.prediction
    
    def getKalmanInterpolation(self, timestamp_increment):
        return self.kf.predict_external(timestamp_increment)
    
    def getLinearInterpolation(self, timestamp):
        if len(self.T) < 2:return None
        filtering_box_1, timestamp_1 = self.T[-1]._get_filtering(), self.T[-1]._get_timestamp()
        filtering_box_2, timestamp_2 = self.T[-2]._get_filtering(), self.T[-2]._get_timestamp()
        ratio_1, ratio_2 = (timestamp - timestamp_1) / (timestamp_2 - timestamp_1), \
                (timestamp_2 - timestamp) / (timestamp_2 - timestamp_1)
        
        interpolation_box = np.array(filtering_box_1) * ratio_2 + np.array(filtering_box_2) * ratio_1
        return interpolation_box
    
    @staticmethod
    def alloc_id():
        Track.count += 1
        np.random.seed(Track.count)
        return Track.count, np.random.randint(0, 255, 3)
    
    def getfeature(self,):
        if self.state == TrackState.Tracked:
            return self.features[1][2]
        else:
            return self.features[2][2]
        
    def re_activate(self, node, new_id = False):
        self.kf.update(node._get_detection())
        self.T.append(node)
        self.update_feature(node)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = node.frame_id
        if new_id:
            self.id, self.c = self.alloc_id()
        
    def mark_lost(self,):
        self.state = TrackState.Lost

    def mark_invalid(self,):
        self.state = TrackState.Invalid

    def mark_removed(self,):
        self.state = TrackState.Removed
        
    def update(self, node):
        if self.id == -1:
            self.id, self.c = self.alloc_id()
        self.frame_id = node._get_frame_id()
        self.T.append(node)
        self.kf.update(node)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.update_feature(node)
        
    def update_feature(self, node):
        if node._get_feature() is None : return
        feature = node._get_feature()
        self.features[0] = [0, 0, feature]
        self.features[1] = self.features[0]
        feat = self.features[2][2]
        smooth_feat = 0.9 * feat + 0.1 * feature
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features[2] = [0, 0, smooth_feat]
        smooth_feat = 0.5 * feat + 0.5 * feature
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features[1] = [0, 0, smooth_feat]
        
    def forward(self, node, res = 0):
        frame_id = node._get_frame_id()
        self.prediction[:2] += res
        node._set_detection(self.prediction)
        self.kf.update(node)
        self.mark_lost()
        if self.state == TrackState.New:
            self.mark_removed()
        elif frame_id - self.frame_id > self.max_time_lost:
            self.mark_invalid()
            
    def getIdColor(self,):
        return self.c if hasattr(self, 'c') else None
        
    

class Tracker:
    """
    Define the compulsory function of a Tracker
    """
    def __init__(self,):
        """
        dim of the reid feature vector
        """
        self.frame_id = 0
        self.New : list[Track] = []
        self.Tracked : list[Track] = []
        self.Lost : list[Track] = []
        self.Invalid : list[Track] = []
        self.Removed : list[Track] = []
        
        self.update_list = []
        
        self.previous_timestamp = 0

    def predict(self, timestamp):
        '''get tracks' prediction'''
        timestamp_increment = self.getInterval(timestamp)
        self.prediction = np.array([t.predict(timestamp_increment) for t in \
                                    self.Tracked + self.Lost + self.New])
            
    def getInterval(self, timestamp):
        interval = timestamp - self.previous_timestamp
        self.previous_timestamp = timestamp
        return interval
            
    def getInterpolation(self, timestamp, method = ['kalman', 'linear'][0]):
        '''
        '''
        timestamp_increment = timestamp - self.previous_timestamp
        interpolation = []
        if method == 'kalman':
            for track in self.Tracked + self.Lost:
                interpolation.append([track.getKalmanInterpolation(timestamp_increment), track.id, track.getIdColor()])
        elif method == 'linear':
            for track in self.Tracked + self.Lost:
                _ret = track.getLinearInterpolation(timestamp)
                if _ret is not None:
                    interpolation.append([_ret, track.id, track.getIdColor()])
        else:
            raise ValueError('unsupported interpolation method')
        return interpolation

    def getCurrentState(self,):
        filtering = []
        for track in self.New + self.Tracked + self.Lost:
            filtering.append([track.getFilteredResults(), track.id, track.getIdColor()])
        return filtering

    def _update(self,
               tracks : List[Track],
               dets : np.ndarray,
               matches : list,
               pkg,
               )->None:
        '''update the updating list, the tracks in update_list are candidate
        tracks, may contain some outliers.'''
        for tid, did in matches:
            track = tracks[tid]
            det = dets[did]
            node = self.createNode(pkg,
                                   det)

            self.update_list.append([track, node])


    def update(self,):
        '''update tracks' state, the update_list are tracks after removing
        outliers'''
        [x[0].update(x[1]) for x in self.update_list]
        self.update_list = []
        
        
    def createNode(self, pkg, det,):
        frame_id = pkg.frame_id
        timestamp = pkg.timestamp
        if det is None:
            feature, detection, score = None, None, None
        else:
            feature, detection, score = det[5:], det[:4], det[4]
        return Node(frame_id, detection, feature, score, timestamp)
    
    def solver(self,
               _pkg : Package)-> None:
        '''tracking solver'''
        
        _lambda = 0.98
        jde_thresh = 0.9
        iou_thresh = 0.9
        
        detection = _pkg.detection.copy()
        detection['bbox'] = xywh2ccwh(detection['bbox'])
        frame = np.concatenate([detection['bbox'],
                                detection['scores'],
                                detection['reid']], axis = -1)
        
        self.predict(_pkg.timestamp)
        
        track_list = self.Tracked + self.Lost + self.New
        
        #print(_pkg.id, _pkg.frame_id)
        
        probes = np.array(list(map(lambda x : x.getfeature(), track_list)))
        
        gallery = frame[:, 5:]
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
        
        '---------------------------second association---------------------'
        tracks = np.array(track_list)[ut]
        
        frame = frame[ud]
        
        iou_matrix = np.zeros((len(ut), len(ud)), dtype = 'float32')
        
        for i, t in enumerate(tracks): iou_matrix[i, :] = t.kf.gating_iou(frame[:, :4].copy())
        
        matches, ut, ud = self.match(1-iou_matrix, iou_thresh)
        
        self._update(tracks, frame, matches, _pkg)
        
        '-------------------forward unassociated tracks-------------------'
        tracks = tracks[ut]
        unassociated_tracks = [t for t in tracks if t.state != TrackState.New]
        for track in unassociated_tracks:track.forward(self.createNode(_pkg, None))
        
        '---------------------update unassociated new tracks--------------'
        new_tracks = [t for t in tracks if t.state == TrackState.New]
        frame = frame[ud]
        iou_matrix = np.zeros((len(new_tracks), len(frame)), dtype = 'float32')
        for i, t in enumerate(new_tracks): 
            iou_matrix[i, :] = t.kf.gating_iou(frame[:, :4].copy())
        matches, ut, ud = self.match(1 - iou_matrix, 0.5)
        self._update(new_tracks, frame, matches, _pkg)
        
        '---------------------------update all candiate tracks-------------'
        self.update()
        
        '---------------------------update states--------------------------'
        TT, LT, IT, RT = [], [], [], []
        
        for t in self.New + self.Tracked + self.Lost:
            if t.state == TrackState.Tracked:TT.append(t)
            elif t.state == TrackState.Lost:LT.append(t)
            elif t.state == TrackState.Invalid:IT.append(t)
            elif t.state == TrackState.Removed:RT.append(t)
            elif t.state == TrackState.New:RT.append(t)
                
        self.Tracked = TT
        self.Lost = LT
        self.Invalid.extend(IT)
        self.Removed.extend(RT)
        self.New = []
        
        '--------------------------generate new tracks---------------------'
        frame = frame[ud]
        
        for detection in frame:
            node = self.createNode(_pkg, detection)
            self.New.append(Track(node))
            
        _pkg._bTracked = True
        _pkg.filtering = self.getCurrentState()
        _pkg._bReadyToPlay = True
        
        
    
    def match(self,
              costs: np.ndarray,
              thresh : float = 0.7)->tuple:
        '''
        costs: cost matrix
        '''
        ut = np.arange(costs.shape[0])
        ud = np.arange(costs.shape[1])
        if costs.size == 0:
            return [], ut, ud

        _, x, y = lap.lapjv(costs,
                            extend_cost = True,
                            cost_limit = thresh)

        matches = []
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])

        ut = np.where(x < 0)[0]
        ud = np.where(y < 0)[0]

        return matches, ut, ud
        
