#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:04:53 2020
@author: hu
"""
import tensorflow as tf
import keras.layers as KL
class DCNv2(KL.Layer):
    def __init__(self, filters, 
                 kernel_size, 
                 #stride, 
                 #padding, 
                 #dilation = 1, 
                 #deformable_groups = 1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        #deformable_groups unsupported
        #dilation unsupported
        #stride unsupported
        #assert stride == 1
        #assert dilation == 1
        #assert deformable_groups == 1
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1, 1, 1)
        #self.padding = padding
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        super(DCNv2, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name = 'kernel',
            shape = self.kernel_size + (int(input_shape[-1]), self.filters),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            trainable = True,
            dtype = 'float32',
            )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = (self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
                )
        
        #[kh, kw, ic, 3 * groups * kh, kw]--->3 * groups * kh * kw = oc [output channels]
        self.offset_kernel = self.add_weight(
            name = 'offset_kernel',
            shape = self.kernel_size + (input_shape[-1], 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]), 
            initializer = 'zeros',
            trainable = True,
            dtype = 'float32')
        
        self.offset_bias = self.add_weight(
            name = 'offset_bias',
            shape = (3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable = True,
            dtype = 'float32',
            )
        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype = 'int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1), tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis = -1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])
        super(DCNv2, self).build(input_shape)
        
        
    def call(self, x):
        #x: [B, H, W, C]
        #offset: [B, H, W, ic] convx [kh, kw, ic, 3 * groups * kh * kw] ---> [B, H, W, 3 * groups * kh * kw]
        offset = tf.nn.conv2d(x, self.offset_kernel, strides = self.stride, padding = 'SAME')
        offset += self.offset_bias
        bs, ih, iw, ic = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        #[B, H, W, 18], [B, H, W, 9]
        oyox, mask = offset[..., :2*self.ks], offset[..., 2*self.ks:]
        mask = tf.nn.sigmoid(mask)
        #[H, W, 2]
        grid_yx = tf.stack(tf.meshgrid(tf.range(iw), tf.range(ih))[::-1], axis = -1)
        #[1, H, W, 9, 2]
        grid_yx = tf.reshape(grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        #[B, H, W, 9, 2]
        grid_yx = tf.cast(grid_yx, 'float32') + tf.reshape(oyox, [bs, ih, iw, -1, 2])
        grid_iy0ix0 = tf.floor(grid_yx)
        ceil = tf.cast(tf.shape(x)[1:3]+1, 'float32')
        grid_iy1ix1 = tf.clip_by_value(grid_iy0ix0 + 1, 0, ceil)
        #[B, H, W, 9, 1] * 2
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis = 4)
        grid_iy0ix0 = tf.clip_by_value(grid_iy0ix0, 0, ceil)
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis = 4)
        grid_yx = tf.clip_by_value(grid_yx, 0, ceil)
        #[B, H, W, 9, 4, 1]
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [1, ih, iw, self.ks, 4, 1])
        #[B, H, W, 9, 4, 2]
        grid = tf.reshape(tf.concat([grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0, grid_ix1, grid_iy0ix0], axis = -1), [bs, ih, iw, self.ks, 4, 2])
        #[B, H, W, 9, 4, 3]
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis = -1)
        #[B, H, W, 9, 2, 2]
        delta = tf.reshape(tf.concat([grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis = -1), [bs, ih, iw, self.ks, 2, 2])
        #[B, H, W, 9, 2, 1] * [B, H, W, 9, 1, 2] = [B, H, W, 9, 2, 2]
        w = tf.expand_dims(delta[..., 0], axis = -1) * tf.expand_dims(delta[..., 1], axis = -2)
        #[B, H+2, W+2, C]
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)], [int(self.pw), int(self.pw)], [0, 0]])
        #[B, H, W, 9, 4, C]
        map_sample = tf.gather_nd(x, grid)
        #([B, H, W, 9, 4, 1] * [B, H, W, 9, 4, C]).SUM(-2) * [B, H, W, 9, 1] = [B, H, W, 9, C]
        map_bilinear = tf.reduce_sum(tf.reshape(w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis = -2) * tf.expand_dims(mask, axis = -1)
        #[B, H, W, 9*C]
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        #[B, H, W, OC]
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides = self.stride, padding = 'SAME')
        if self.use_bias:
            output += self.bias
        return output
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)