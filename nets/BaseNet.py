#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:38:21 2020

@author: hu
"""
import tensorflow as tf
from functools import wraps
from functools import reduce
import keras.layers as KL
import keras.backend as K
from keras.regularizers import l1_l2
l1l2_reg = l1_l2(l1 = 0., l2 = 5e-4)
reg = [None, None]

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def mconv_block(inputs, filters, stage, block, strides=(1, 1), set_shortcut = False, use_bias=True, training=None):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = DarknetConv2D1(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(inputs)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=training)
    x = KL.Activation('relu')(x)

    x = DarknetConv2D1(nb_filter2, 3, padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=training)
    x = KL.Activation('relu')(x)

    x = DarknetConv2D1(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=training)

    if set_shortcut:
        shortcut = DarknetConv2D1(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(inputs)
        shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=training)
    else:
        shortcut = inputs
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x
def conv_block(inputs, filters, name, set_shortcut = False, strides = (1, 1), training = None):
    n_f1, n_f2, n_f3 = filters
    x = DarknetConv2D_BN_Relu(n_f1, 1, strides = (1, 1), name = name + 'dlt1', training = training)(inputs)
    x = DWConv2D_BN_Relu(n_f2, 1, strides = strides, padding = 'same', name = name + 'dlt2', training = training)(x)
    x = DarknetConv2D1(n_f3, 1, use_bias = False, strides = (1, 1), name = name + 'dct1')(x)
    if set_shortcut:
        shortcut = DarknetConv2D1(n_f3, 1, use_bias = False, strides = strides, name = name + 'dct2')(inputs)
    else:
        shortcut = inputs
    x = KL.Add()([x, shortcut])
    x = KL.BatchNormalization(name = name + 'bn1')(x, training = training)
    x = KL.Activation('relu')(x)
    return x

def basicUnit(x, name, strides = 1, training = None):
    x = conv_block(x, [256, 256, 1024], name = name + str(1), set_shortcut = True, training = training)
    x = conv_block(x, [256, 256, 1024], name = name + str(2), training = training)
    #x = conv_block(x, [512, 512, 2048], name = name + str(4), strides = strides, set_shortcut = True, training = training)
    #x = conv_block(x, [512, 512, 2048], name = name + str(5), training = training)
    return x

def gatherValidItems(inputs, ids,):
    x = KL.Lambda(lambda x : tf.reshape(x, [-1, x.get_shape()[-3], x.get_shape()[-2], x.get_shape()[-1]]))(inputs)
    ids = KL.Lambda(lambda x : tf.cast(tf.reshape(x, [-1]), 'int32'))(ids)
    x = KL.Lambda(lambda x : tf.gather(x[0], tf.reshape(tf.where(tf.not_equal(x[1], -1)), [-1])))([x, ids])
    return x


def sMGN(inputs, _eval, training = None, return_all = True, return_mgn = False, l2_norm = True):
    reg[0] = l1l2_reg
    b2 = basicUnit(inputs, name = 'my_bu2', training = training)
    print(b2.shape)
    shape = [b2.get_shape()[1].value, b2.get_shape()[2].value]
    b2_g_max = KL.MaxPooling2D((shape[0], shape[1]))(b2)
    b2_g_ave = KL.AveragePooling2D((shape[0], shape[1]))(b2)
    b2_g = KL.Concatenate(axis = -1)([b2_g_max, b2_g_ave])
    
    b2_s_max = KL.MaxPooling2D((shape[0] // 2, shape[1]), strides = (shape[0] // 2, shape[1]), padding = 'valid')(b2)
    b2_s_ave = KL.AveragePooling2D((shape[0] // 2, shape[1]), strides = (shape[0] // 2, shape[1]), padding = 'valid')(b2)
    b2_s = KL.Concatenate(axis = -1)([b2_s_max, b2_s_ave])
    
    b3_s_max = KL.MaxPooling2D((shape[0] // 3, shape[1]), strides = (shape[0] // 3, shape[1]), padding = 'valid')(b2)
    b3_s_ave = KL.AveragePooling2D((shape[0] // 3, shape[1]), strides = (shape[0] // 3, shape[1]), padding = 'valid')(b2)
    b3_s = KL.Concatenate(axis = -1)([b3_s_max, b3_s_ave])
    
    b2_s1, b2_s2 = KL.Lambda(lambda x : [x[:, 0:1, :, :], x[:, 1:2, :, :]])(b2_s)
    b3_s1, b3_s2, b3_s3 = KL.Lambda(lambda x : [x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]])(b3_s)
    reg[1] = None
    if return_all:
        b_all = [DarknetConv2D_BN_Linear(256, 1, strides = (2, 2), name = 'my_dlt' + str(i), training = training)(x) for i, x in enumerate([b2_g, b2_s1, b2_s2, b3_s1, b3_s2, b3_s3])]
        if return_mgn:
            return KL.Lambda(lambda x : tf.nn.l2_normalize(tf.concat(x, axis = -1), dim = -1))(b_all)
        b_all = KL.Lambda(lambda x : x, name = 'l2_norm')(b_all)
        b_all = KL.Lambda(lambda x : [tf.nn.l2_normalize(f, dim = -1) for f in x], name = 'l2_norm')(b_all) if _eval else b_all
        return KL.Lambda(lambda x : tf.concat(x, axis = -1), )(b_all)
    else:
        bg = DarknetConv2D_BN_Linear(256, 1, strides = (2, 2), name = 'my_dlt' + str(0), training = training)(b2_g)
        return KL.Lambda(lambda x : tf.nn.l2_normalize(x, dim = -1))(bg) if _eval else bg

'''
def sMGN(inputs, _eval, training = None, return_all = True, return_mgn = False, l2_norm = True):
    reg[0] = l1l2_reg
    b2 = basicUnit(inputs, name = 'my_bu2', training = training)
    shape = b2._keras_shape[-3:-1]
    b2_g_max = KL.MaxPooling2D((shape[0], shape[1]))(b2)
    b2_g_ave = KL.AveragePooling2D((shape[0], shape[1]))(b2)
    
    b2_s_max = KL.MaxPooling2D((shape[0] // 2, shape[1]), strides = (shape[0] // 2, shape[1]), padding = 'valid')(b2)
    b2_s_ave = KL.AveragePooling2D((shape[0] // 2, shape[1]), strides = (shape[0] // 2, shape[1]), padding = 'valid')(b2)
    b2_s1_max, b2_s2_max = KL.Lambda(lambda x : [x[:, 0:1, :, :], x[:, 1:2, :, :]])(b2_s_max)
    b2_s1_ave, b2_s2_ave = KL.Lambda(lambda x : [x[:, 0:1, :, :], x[:, 1:2, :, :]])(b2_s_ave)
    
    b3_s_max = KL.MaxPooling2D((shape[0] // 3, shape[1]), strides = (shape[0] // 3, shape[1]), padding = 'valid')(b2)
    b3_s_ave = KL.AveragePooling2D((shape[0] // 3, shape[1]), strides = (shape[0] // 3, shape[1]), padding = 'valid')(b2)
    b3_s1_max, b3_s2_max, b3_s3_max = KL.Lambda(lambda x : [x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]])(b3_s_max)
    b3_s1_ave, b3_s2_ave, b3_s3_ave = KL.Lambda(lambda x : [x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]])(b3_s_ave)
    
    b_all_max = [b2_g_max, b2_s1_max, b2_s2_max, b3_s1_max, b3_s2_max, b3_s3_max]
    b_all_ave = [b2_g_ave, b2_s1_ave, b2_s2_ave, b3_s1_ave, b3_s2_ave, b3_s3_ave]
    
    reg[1] = None
    if return_all:
        b_all_max = [DarknetConv2D_BN_Linear(192, 1, strides=(2, 2), name = 'my_dlt_max_%d'%i, training = training)(x) for i, x in enumerate(b_all_max)]
        b_all_ave = [DarknetConv2D_BN_Linear(64, 1, strides=(2, 2), name = 'my_dlt_ave_%d'%i, training = training)(x) for i, x in enumerate(b_all_ave)]
        b_all = [KL.Lambda(lambda x : tf.concat([x[0], x[1]], axis = -1))([f1, f2]) for f1, f2 in list(zip(b_all_max, b_all_ave))]
        if return_mgn:
            return KL.Lambda(lambda x : tf.nn.l2_normalize(tf.concat(x, axis = -1), dim = -1))(b_all)
        b_all = KL.Lambda(lambda x : x, name = 'l2_norm')(b_all)
        b_all = KL.Lambda(lambda x : [tf.nn.l2_normalize(f, dim = -1) for f in x], name = 'l2_norm')(b_all) if _eval else b_all
        return KL.Lambda(lambda x : tf.concat(x, axis = -1), )(b_all)
    else:
        bg_max = DarknetConv2D_BN_Linear(192, 1, strides = (2, 2), name = 'my_dlt_max' + str(0), training = training)(b2_g_max)
        bg_ave = DarknetConv2D_BN_Linear(64, 1, strides = (2, 2), name = 'my_dlt_ave' + str(0), training = training)(b2_g_ave)
        bg = KL.Lambda(lambda x : tf.concat(x, axis = -1))([bg_max, bg_ave])
        return KL.Lambda(lambda x : tf.nn.l2_normalize(x, dim = -1))(bg) if _eval else bg
'''
@wraps(KL.Conv2D)
def DarknetConv2D1(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': reg[0], 'activity_regularizer':reg[1]}
    darknet_conv_kwargs.update(kwargs)
    return KL.Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Linear(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    assert 'training' in kwargs
    del kwargs['training']
    no_bias_kwargs = {'use_bias': False}
    name = kwargs.get('name')
    if name is None:
        names = [None, None]
    else:
        names = [name + '_conv', name + '_bn']
    kwargs['name'] = names[0]
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D1(*args, **no_bias_kwargs),
        KL.BatchNormalization(name = names[1]))

def DarknetConv2D_BN_Logistic(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    assert 'training' in kwargs
    del kwargs['training']
    no_bias_kwargs = {'use_bias': False}
    name = kwargs.get('name')
    if name is None:
        names = [None, None]
    else:
        names = [name + '_conv', name + '_bn']
    kwargs['name'] = names[0]
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D1(*args, **no_bias_kwargs),
        KL.BatchNormalization(name = names[1]),
        KL.Activation('sigmoid'))
    
def DWConv2D_BN_Relu(*args, **kwargs):
    assert 'training' in kwargs
    del kwargs['training']
    no_bias_kwargs = {'use_bias': False}
    name = kwargs.get('name')
    if name is None:
        names = [None, None, None]
    else:
        names = [name + '_conv', name + '_dwconv', name + '_bn']
    kwargs['name'] = names[0]
    no_bias_kwargs.update(kwargs)
    return compose(
            DarknetConv2D1(*args, **no_bias_kwargs),
            KL.DepthwiseConv2D(3, padding = 'same', use_bias = False, kernel_regularizer = reg[0], name = names[1]),
            KL.BatchNormalization(name = names[2]),
            KL.Activation('relu'))
    
def DarknetConv2D_BN_Relu(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    assert 'training' in kwargs
    del kwargs['training']
    no_bias_kwargs = {'use_bias': False}
    name = kwargs.get('name')
    if name is None:
        names = [None, None]
    else:
        names = [name + '_conv', name + '_bn']
    kwargs['name'] = names[0]
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D1(*args, **no_bias_kwargs),
        KL.BatchNormalization(name = names[1]),
        KL.Activation('relu'))
    

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

@wraps(KL.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': reg[0]}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return KL.Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    assert 'training' in kwargs
    training = kwargs['training']
    del kwargs['training']
    no_bias_kwargs = {'use_bias': False}
    name = kwargs.get('name')
    if name is None:
        names = [None, None]
    else:
        names = [name + '_conv', name + '_bn']
    kwargs['name'] = names[0]
    no_bias_kwargs.update(kwargs)

    conv2d = DarknetConv2D(*args, **no_bias_kwargs)
    bn = KL.BatchNormalization(name = names[1])
    l = KL.LeakyReLU(alpha=0.1)
    
    return lambda x : l(bn(conv2d(x), training = training))

def spp(spp = False, name = None, training = None):
    if not spp:return lambda x : x
    res = DarknetConv2D_BN_Leaky(512, (1, 1), name = name, training = training)
    return lambda x : res(\
             KL.Concatenate()([KL.MaxPooling2D(pool_size=13, strides=1, padding='same')(x),
             KL.MaxPooling2D(pool_size=9, strides=1, padding='same')(x),
             KL.MaxPooling2D(pool_size=5, strides=1, padding='same')(x),
             x]))

class Mish(KL.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    assert 'training' in kwargs
    training = kwargs['training']
    del kwargs['training']

    no_bias_kwargs = {'use_bias': False}
    name = kwargs.get('name')
    if name is None:
        names = [None, None]
    else:
        names = [name + '_conv', name + '_bn']
    kwargs['name'] = names[0]
    no_bias_kwargs.update(kwargs)
    
    conv2d = DarknetConv2D(*args, **no_bias_kwargs)
    bn = KL.BatchNormalization(name = names[1])
    l = Mish()
    return lambda x : l(bn(conv2d(x), training = training))   

def LookWhat(inputs, name):
    '''Squeeze-and-Excitation net'''
    x = KL.GlobalAveragePooling2D()(inputs)
    x = KL.Lambda(lambda x : tf.expand_dims(tf.expand_dims(x, axis = 1), axis = 1))(x)
    x = KL.Conv2D(inputs.get_shape()[-1].value//16, 1, activation = 'relu', name = name + '1')(x)
    x = KL.Conv2D(inputs.get_shape()[-1].value, 1, activation = 'sigmoid', name = name + '2')(x)
    x = KL.Lambda(lambda x : x[0] * x[1], name = name + '_vis_lambda')([inputs, x])
    return x

def LookWhere(inputs, name):
    '''spatial-attention'''
    mask = KL.Conv2D(1, 1, strides = 1, name = name, use_bias = False, activation = 'sigmoid', kernel_initializer = 'zeros')(inputs)
    return KL.Lambda(lambda x : x[0] * x[1])([inputs, mask])

def ATLnet(inputs, layer = 0, SEnet = False):
    reg[0] = l1l2_reg
    reid_map = inputs[layer]
    reid_map = LookWhat(reid_map, name = 'ALTnet_LookWhat')
    #reid_map = LookWhere(reid_map, name = 'ALTnet_LookWhere')
    dim = 256 if reid_map.get_shape()[-1].value <= 256 else reid_map.get_shape()[-1].value
    reid_map = DarknetConv2D1(dim, 1, strides = (1, 1), name = 'yfm_dc3')(reid_map)
    reid_map = KL.DepthwiseConv2D(3, padding = 'same', kernel_regularizer = reg[0], name = 'yfm_dwc4')(reid_map)
    return [reid_map]

class LookUpTable(KL.Layer):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.table_size = cfg.TABLE_SIZE
        self.max_volume = cfg.max_volume
        self.updaterate = 0.5
        self.inv_tao = 15
        self.droprate = 0.5
        self.dynamic_table_size = cfg.DTABLE_SIZE
        self.fl = cfg.FL
        self.fn = cfg.FN
        self.l2_norm = cfg.l2_norm
        self.improve = False
        self.sim_type = cfg.sim_type
        self.gallery_binit = tf.Variable(tf.zeros((self.table_size, self.max_volume + 1), dtype = 'float32'), trainable = False)
        self.binit = tf.Variable(tf.zeros((self.table_size,), dtype = 'float32'), trainable = False)
        self.gallery_table = tf.Variable(tf.zeros((self.table_size, self.max_volume + 1, self.fl * self.fn)), trainable = False)
        self.table = tf.Variable(tf.zeros((self.table_size, self.fl * self.fn)), trainable = False)
        self.train_CQ = tf.Variable(tf.zeros((self.dynamic_table_size, self.fl * self.fn)), dtype = 'float32', trainable = False)
        self.test_CQ = tf.Variable(tf.zeros((self.dynamic_table_size, self.fl * self.fn)), dtype = 'float32', trainable = False)
        self.bTrain = tf.cast(tf.Variable(cfg.TRAIN_LIST, trainable = False), dtype = 'float32')
        super(LookUpTable, self).__init__(**kwargs)
        
    def getSimilaritiesAndLoss(self, features, table_features_active, table_features_inactive, indices, bTrain = True, bLoss = True, bDrop = False):
        #[N11, N2, fdim]
        table_features = tf.concat([table_features_active, table_features_inactive], axis = 0)
        train_CQ = tf.stop_gradient(self.train_CQ)
        test_CQ = tf.stop_gradient(self.test_CQ)
        if bTrain:
            table_features = tf.concat([table_features, train_CQ[-((self.dynamic_table_size * 3)//4):, :]], axis = 0)
        else:
            table_features = tf.concat([table_features, test_CQ[-((self.dynamic_table_size * 3)//4):, :]], axis = 0)

        feats = tf.reshape(features, [tf.shape(features)[0], self.fn, self.fl])
        table = tf.reshape(table_features, [tf.shape(table_features)[0], self.fn, self.fl])
        feats_norm = tf.nn.l2_normalize(feats, axis = -1)
        table_norm = tf.nn.l2_normalize(table, axis = -1)

        if self.sim_type == 'cosine':
            similarities = tf.reduce_sum(tf.expand_dims(feats_norm, axis = 1) * tf.expand_dims(table_norm, axis = 0), axis = -1)
        elif self.sim_type == 'l2norm_cosine':
            cosine = tf.reduce_sum(tf.expand_dims(feats_norm, axis = 1) * tf.expand_dims(table_norm, axis = 0), axis = -1)
            l2norm = 1 - tf.norm(tf.clip_by_value(tf.abs(tf.expand_dims(feats_norm, axis = 1) - tf.expand_dims(table_norm, axis = 0)), 1e-5, 1e5), axis = -1)
            similarities = 0.5 * (cosine + l2norm)
        elif self.sim_type == 'l2norm':
            feats_norm = tf.tile(tf.expand_dims(tf.norm(feats, axis = -1), axis = 1), [1, tf.shape(table_features)[0], 1])
            table_norm = tf.tile(tf.expand_dims(tf.norm(table, axis = -1), axis = 0), [tf.shape(features)[0], 1, 1])
            max_ = tf.maximum(feats_norm, table_norm)
            similarities = 1 - tf.norm(tf.clip_by_value(tf.abs(tf.expand_dims(feats, axis = 1) - tf.expand_dims(table, axis = 0)), 1e-5, 1e5), axis = -1) / max_
        elif self.sim_type == 'theta':
            similarities = 1 - 2 * (tf.acos(tf.clip_by_value(tf.reduce_sum(tf.expand_dims(feats_norm, axis = 1) * tf.expand_dims(table_norm, axis = 0), axis = -1), -1 + 1e-5, 1 - 1e-5)) / 3.1416)
        else:
            raise ValueError('unsupported similairity type')
            
        n_items = tf.clip_by_value(tf.cast(tf.shape(indices)[0], tf.float32), 1, 1e10)
        similarities = tf.split(similarities, self.fn, axis = 2)
        softmax_similarities = [tf.nn.softmax(tf.concat([s[:, :, 0], tf.ones([tf.shape(s)[0], 1], 'float32') * -1.0], axis = -1) * self.inv_tao, dim = -1) for s in similarities]
        
        if bLoss:
            loss_list = [tf.reduce_sum(-tf.log(tf.gather_nd(s, indices))) / n_items for s in softmax_similarities]
            loss = tf.tile(tf.reshape(tf.reduce_mean(tf.stack(loss_list)), [1, -1]), [self.cfg.BATCH_SIZE, 1])
            return softmax_similarities, loss
        else:
            return softmax_similarities

    def getIndices(self, features, ids, bTrain = True):
        if bTrain:
            ids_indices = tf.reshape(tf.where(tf.greater(tf.reshape(tf.gather(self.bTrain, ids), [-1]), 0.5)), [-1])
            mask0 = tf.cast(tf.reshape(tf.where(tf.logical_and(tf.greater(self.binit, 0.5), tf.greater(self.bTrain, 0.5))), [-1]), tf.int32)
        else:
            ids_indices = tf.reshape(tf.where(tf.less(tf.reshape(tf.gather(self.bTrain, ids), [-1]), 0.5)), [-1])
            mask0 = tf.cast(tf.reshape(tf.where(tf.logical_and(tf.greater(self.binit, 0.5), tf.less(self.bTrain, 0.5))), [-1]), tf.int32)
        features_new = tf.gather(features, ids_indices)
        ids_new = tf.gather(ids, ids_indices)
        location_id = tf.expand_dims(ids_new, axis = 1) - tf.expand_dims(mask0, axis = 0)
        mask1 = tf.gather(mask0, tf.reshape(tf.where(tf.equal(tf.reduce_min(tf.abs(location_id), axis = 0), 0)), [-1]))
        mask2 = tf.random_shuffle(tf.gather(mask0, tf.reshape(tf.where(tf.greater(tf.reduce_min(tf.abs(location_id), axis = 0), 0)), [-1])))[:(self.dynamic_table_size//4)]
        mask3 = tf.concat([mask1, mask2], axis =0)
        
        feature_table_active = tf.gather(self.table, mask1)
        feature_table_inactive = tf.gather(self.table, mask2)
        
        location_id = tf.expand_dims(ids_new, axis = 1) - tf.expand_dims(mask3, axis = 0)
        indices = tf.where(tf.equal(location_id, 0))
        return location_id, indices, features_new, feature_table_active, feature_table_inactive

    def getAcc(self, location_id, sm, threshold = 0.0):
        l_id = tf.concat([location_id, tf.ones([tf.shape(location_id)[0], 1], dtype = tf.int32)], axis = -1)
        pred = [tf.reshape(tf.clip_by_value(tf.cast(tf.argmax(s, axis = -1), 'int32'), 0, tf.shape(l_id)[-1] - 1), [-1, 1]) for s in sm]
        ind = tf.cast(tf.reshape(K.arange(0, stop = tf.shape(pred[0])[0]), [-1, 1]), tf.int32)
        indices = [tf.concat([ind, p], axis = -1) for p in pred]
        values = [tf.gather_nd(l_id, idx) for idx in indices]
        amps = [tf.gather_nd(s, idx) for s, idx in list(zip(sm, indices))]
        accs = [tf.cast(tf.shape(tf.where(tf.logical_and(tf.equal(val, 0), tf.greater(amp, threshold))))[0], tf.float32) / tf.clip_by_value(tf.cast(tf.shape(l_id)[0], tf.float32), 1, 1e10) for val, amp in list(zip(values, amps))]
        accs = tf.concat([tf.tile(tf.reshape(acc, [1, -1]), [self.cfg.BATCH_SIZE, 1]) for acc in accs], axis = -1)
        return accs

    def updateCQ(self, train_features, test_features):
        train_CQ = tf.concat([self.train_CQ, train_features], axis = 0)[-self.dynamic_table_size:]
        self.train_CQ = tf.scatter_update(self.train_CQ, K.arange(0, stop = self.dynamic_table_size), train_CQ)
        test_CQ = tf.concat([self.test_CQ, test_features], axis = 0)[-self.dynamic_table_size:]
        self.test_CQ = tf.scatter_update(self.test_CQ, K.arange(0, stop = self.dynamic_table_size), test_CQ)

    def getTable(self, ids, loc, features):
        '''
        ids:[N]
        loc:[N]
        '''
        #indices: [N, 2]
        loc = tf.clip_by_value(loc, 0, self.max_volume)
        indices = tf.stack([tf.range(0, limit = tf.shape(ids)[0], dtype = 'int32'), loc], axis = 1)
        gallery_binit = tf.gather(self.gallery_binit, ids)
        exclude_gallery_binit = tf.tensor_scatter_nd_update(gallery_binit, indices, tf.zeros_like(ids, dtype = 'float32'))
        #[N, V, dims]
        gallery_table = tf.gather(self.gallery_table, ids)
        exclude_feature_table = gallery_table[:, :self.max_volume, :] * tf.expand_dims(exclude_gallery_binit[:, :self.max_volume], axis = 2)
        #[N, dims]
        feature_table = tf.reduce_sum(exclude_feature_table, axis = 1) / tf.clip_by_value(tf.reduce_sum(exclude_gallery_binit, axis = 1, keepdims = True), 1, 1e10)
        self.table = tf.scatter_update(self.table, ids, feature_table)
        self.binit = tf.scatter_update(self.binit, ids, tf.reduce_max(exclude_gallery_binit[:, :self.max_volume], axis = 1))
        
        indices = tf.stack([ids, loc], axis = 1)
        self.gallery_table = tf.scatter_nd_update(self.gallery_table, indices, features)
        self.gallery_binit = tf.scatter_nd_update(self.gallery_binit, indices, tf.ones_like(ids, dtype = 'float32'))

    def updateTable(self, ids, features):
        init = tf.cast(tf.reshape(tf.gather(self.binit, ids), [-1, 1]), tf.float32)
        u_features = tf.gather(self.table, ids)
        u_features = features * (1 - init * (1 - self.updaterate)) + u_features * (init * (1 - self.updaterate))
        self.table = tf.scatter_update(self.table, ids, u_features)
        self.binit = tf.scatter_update(self.binit, ids, tf.ones_like(ids, dtype = tf.float32))

    def updateGalleryTable(self, ids):
        gallery_binit = tf.gather(self.gallery_binit, ids)
        gallery_table = tf.gather(self.gallery_table, ids)
        feature_table = tf.reduce_sum(gallery_table[:, :self.max_volume, :], axis = 1) / tf.clip_by_value(tf.reduce_sum(gallery_binit[:, :self.max_volume], axis = 1, keepdims = True), 1, 1e10)
        self.table = tf.scatter_update(self.table, ids, feature_table)
        self.binit = tf.scatter_update(self.binit, ids, tf.ones_like(ids, dtype = tf.float32))

    def call(self, inputs):
        '''
        inputs: [B, N1, fdim]
        ids: [B, N1, id]
        self.table: [table_size, fdim]
        '''
        features = inputs[0]
        
        #ids: [B * N1]
        ids = inputs[1]
        ids = tf.cast(tf.reshape(ids, [-1]), 'int32')
        #loc: [B * N1]
        loc = inputs[2]
        loc = tf.cast(tf.reshape(loc, [-1]), 'int32')
        #[N11]
        indices = tf.reshape(tf.where(tf.not_equal(ids, -1)), [-1])
        ids = tf.gather(ids, indices)
        loc = tf.gather(loc, indices)
        
        valid_indices = tf.reshape(tf.where(tf.greater_equal(ids, 0)), [-1])
        invalid_train_indices = tf.reshape(tf.where(tf.equal(ids, -2)), [-1])
        invalid_test_indices = tf.reshape(tf.where(tf.equal(ids, -3)), [-1])
        #[N11]
        valid_ids = tf.gather(ids, valid_indices)
        valid_loc = tf.gather(loc, valid_indices)
        valid_features = tf.gather(features, valid_indices)
        
        if self.improve:
            self.getTable(valid_ids, valid_loc, valid_features)
        
        invalid_train_features = tf.gather(features, invalid_train_indices)
        invalid_test_features = tf.gather(features, invalid_test_indices)

        self.updateCQ(invalid_train_features, invalid_test_features)
        train_CQ = tf.stop_gradient(self.train_CQ)
        test_CQ = tf.stop_gradient(self.test_CQ)
        #train ids
        location_id_train, indices_train, features_train, features_table_train_active, features_table_train_inactive = self.getIndices(valid_features, valid_ids, bTrain = True)
        location_id_test, indices_test, features_test, features_table_test_active, features_table_test_inactive = self.getIndices(valid_features, valid_ids, bTrain = False)
        positive_sm, train_loss = self.getSimilaritiesAndLoss(features_train, features_table_train_active, features_table_train_inactive, indices_train, bTrain = True, bLoss = True, bDrop = False)
        test_sm, test_loss = self.getSimilaritiesAndLoss(features_test, features_table_test_active, features_table_test_inactive, indices_test, bTrain = False, bLoss = True)

        #get accuracy
        train_accs = self.getAcc(location_id_train, positive_sm)
        test_accs = self.getAcc(location_id_test, test_sm)
        
        #update feature table
        if self.improve:
            self.updateGalleryTable(valid_ids)
        else:
            self.updateTable(valid_ids, valid_features)

        #set up monitor to keep the variables connect with network graph
        gallery_table = tf.stop_gradient(self.gallery_table)
        gallery_table = tf.cast(tf.tile(tf.reshape(tf.reduce_sum(gallery_table), [1, -1]), [self.cfg.BATCH_SIZE, 1]), dtype = tf.float32)
        gallery_binit = tf.stop_gradient(self.gallery_binit)
        gallery_ninit = tf.cast(tf.tile(tf.reshape(tf.reduce_sum(gallery_binit[:, :self.max_volume]), [1, -1]), [self.cfg.BATCH_SIZE, 1]), dtype = tf.float32)
        
        table = tf.stop_gradient(self.table)
        table = tf.cast(tf.tile(tf.reshape(tf.reduce_sum(table), [1, -1]), [self.cfg.BATCH_SIZE, 1]), dtype = tf.float32)
        binit = tf.stop_gradient(self.binit)
        ninit = tf.cast(tf.tile(tf.reshape(tf.reduce_sum(binit), [1, -1]), [self.cfg.BATCH_SIZE, 1]), dtype = tf.float32)

        monitor1 = tf.cast(tf.tile(tf.reshape(tf.reduce_sum(tf.cast(tf.not_equal(test_CQ, 0), 'float32')), [1, -1]), [self.cfg.BATCH_SIZE, 1]), dtype = tf.float32)
        monitor2 = tf.cast(tf.tile(tf.reshape(tf.reduce_sum(tf.cast(tf.not_equal(train_CQ, 0), 'float32')), [1, -1]), [self.cfg.BATCH_SIZE, 1]), dtype = tf.float32)

        loss = tf.concat([train_loss, train_accs[:, :1], test_loss, test_accs[:, :1], gallery_ninit, gallery_table, ninit, table, monitor1, monitor2], axis = -1)
        return loss
    def compute_output_shape(self, input_shape):
        return tuple([self.cfg.BATCH_SIZE, 10])


class feature_pooling(KL.Layer):
    '''aligned roi pooling'''
    def __init__(self, config, **kwargs):
        super(feature_pooling, self).__init__(**kwargs)
        self.config = config
    
    def call(self, inputs):
        bboxes = inputs[0][..., :4]
        feature_maps = inputs[1:]
        boxes = tf.reshape(bboxes, [tf.shape(bboxes)[0] * tf.shape(bboxes)[1], tf.shape(bboxes)[2]])
        box_indices = tf.reshape(tf.transpose(tf.tile(tf.reshape(tf.range(tf.shape(bboxes)[0]), [1, -1]), [tf.shape(bboxes)[1], 1])), [-1])
        pooled = tf.image.crop_and_resize(feature_maps[0], boxes, box_indices, self.config.POOL_KERNEL,method="bilinear")
        return tf.reshape(pooled, [tf.shape(bboxes)[0], tf.shape(bboxes)[1], *self.config.POOL_KERNEL, pooled.get_shape()[-1].value])
        
    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + tuple([*self.config.POOL_KERNEL, input_shape[1][-1]])
    
class CropAndResize(KL.Layer):
    '''aligned roi pooling'''
    def __init__(self, size, **kwargs):
        super(CropAndResize, self).__init__(**kwargs)
        self.size = size
    
    def call(self, inputs):
        bboxes = inputs[0][..., :4]
        feature_maps = inputs[1:]
        boxes = tf.reshape(bboxes, [tf.shape(bboxes)[0] * tf.shape(bboxes)[1], tf.shape(bboxes)[2]])
        box_indices = tf.reshape(tf.transpose(tf.tile(tf.reshape(tf.range(tf.shape(bboxes)[0]), [1, -1]), [tf.shape(bboxes)[1], 1])), [-1])
        pooled = tf.image.crop_and_resize(feature_maps[0], boxes, box_indices, self.size,method="bilinear")
        return tf.reshape(pooled, [tf.shape(bboxes)[0], tf.shape(bboxes)[1], *self.size, pooled.get_shape()[-1].value])
        
    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + tuple([*self.size, input_shape[1][-1]])