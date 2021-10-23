#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:50:00 2020

@author: hu
"""
import keras.backend as K
import h5py
import numpy as np
import os
import sys
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curpath)
from saving import load_attributes_from_hdf5_group, preprocess_weights_for_loading
sys.path.remove(curpath)

def load_weights(weights_dict, layer,
                 weight_value_tuples,
                 reshape=False
                 ):
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            load_weights(weights_dict, l, 
                         weight_value_tuples = weight_value_tuples,
                         reshape = reshape)
    else:
        weights = []
        for i in range(len(layer.weights)):
            if layer.weights[i].name in weights_dict:
                weights.append(weights_dict[layer.weights[i].name])
            else:
                print("can not load weights for layer %s."%(layer.weights[i].name))
        if len(weights) > 0:
            weights = preprocess_weights_for_loading(layer, weights)
            for j in range(len(weights)):
                #print(layer.weights[j].name, layer.weights[j].shape, weights[j].shape)
                weight_value_tuples.append((layer.weights[j], weights[j]))

def compare_weights(path1, path2):
    f = h5py.File(path1)
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    weights_dict1 = {}
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        [weights_dict1.update({weight_name : np.asarray(g[weight_name])}) for weight_name in weight_names]
    
    f = h5py.File(path2)
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    weights_dict2 = {}
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        [weights_dict2.update({weight_name : np.asarray(g[weight_name])}) for weight_name in weight_names]
    print(weights_dict1.keys())
    print(weights_dict2.keys())
    keys = list(set(weights_dict1.keys()).intersection(list(weights_dict2.keys())))
    print(keys)
    for key in keys:
        if (weights_dict1[key] == weights_dict2[key]).all():
            print(key, 'same')
            continue
        print(key, 'diff')
    print('fini...')

def load_weights_by_name_from_weights_dict(model, weights_dict, reshape = False):
    layers = model.layers
    weight_value_tuples = []
    for layer in layers:
        load_weights(weights_dict, 
                     layer, 
                     weight_value_tuples,
                     reshape=reshape)
    
    K.batch_set_value(weight_value_tuples)

def load_weights_by_name(model, path, reshape=False):
    f = h5py.File(path, 'r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    

    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    weights_dict = {}
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        [weights_dict.update({weight_name : np.asarray(g[weight_name])}) for weight_name in weight_names]
    
    load_weights_by_name_from_weights_dict(model, weights_dict)

    
