import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import keras.backend as K
from .GroupNormalization import InstanceNormalization
import torch
import numpy as np
class keras_osnet:
    '''a keras implementation of osnet_ain'''
    def __init__(self,):
        self.blocks = [[self.OSBlockINin, self.OSBlockINin],
                       [self.OSBlock, self.OSBlockINin],
                       [self.OSBlockINin, self.OSBlock]]
        self.feature_dim = 512
        self.layers = [2, 2, 2]
        self.channels = [64, 256, 384, 512]
        self.conv1_IN = True
        
    def ConvLayer(self, x, IN, name, **kwargs):
        x = KL.Conv2D(name = name + '.conv',**kwargs)(x)
        if IN:
            x = InstanceNormalization(
                                   epsilon = 1e-5,
                                   name = name + '.bn')(x)
        else:
            x = KL.BatchNormalization(epsilon = 1e-5,
                                      name = name + '.bn')(x)
        x = KL.Activation('relu')(x)
        return x

    def Conv1x1(self, x, filters, name, strides = 1):
        x = KL.Conv2D(filters = filters,
                      kernel_size = (1, 1),
                      strides = strides,
                      padding = 'valid',
                      name = name + '.conv',
                      use_bias = False,)(x)
        x = KL.BatchNormalization(epsilon = 1e-5,
                                  name = name + '.bn')(x)
        x = KL.Activation('relu')(x)
        
        return x
    
    def Conv1x1Linear(self, x, filters, name, strides = 1, bn = True):
        x = KL.Conv2D(filters = filters,
                      kernel_size = 1,
                      strides = strides,
                      padding = 'valid',
                      use_bias = False,
                      name = name + '.conv',
                      )(x)
        if bn:
            x = KL.BatchNormalization(epsilon = 1e-5,
                                      name = name + '.bn')(x)
        return x
    
    def Conv3x3(self, x, filters, strides = 1):
        if strides == (2, 2) or strides == 2:
            x = KL.ZeroPadding2D((1, 1))(x)
            padding = 'valid'
        else:
            padding = 'same'
        x = KL.Conv2D(filters = filters,
                      kernel_size = (3, 3),
                      strides = strides,
                      padding = padding,
                      use_bias = False,)
        
        x = KL.BatchNormalization(epsilon = 1e-5)(x)
        x = KL.Activation('relu')(x)
        return x
    
    def LightConv3x3(self, x, filters, name):
        x = KL.Conv2D(filters = filters,
                      kernel_size = (1, 1),
                      strides = 1,
                      padding = 'valid',
                      use_bias = False,
                      name = name + '.conv1'
                      )(x)
        
        x = KL.DepthwiseConv2D(kernel_size = (3, 3),
                               strides = 1,
                               padding = 'same',
                               use_bias = False,
                               name = name + '.conv2')(x)
        
        x = KL.BatchNormalization(epsilon = 1e-5,
                                  name = name + '.bn')(x)
        
        x = KL.Activation('relu')(x)
        
        return x
    
    def LightConvStream(self, x, filters, depth, name):
        assert depth >= 1
        for i in range(depth):
            x = self.LightConv3x3(x, filters, name = name + '.layers.%d'%i)
        return x
    
    def ChannelGate(self,
                    dims,
                    name,
                    num_gates = None,
                    return_gates = False,
                    gate_activation = 'sigmoid',
                    reduction = 16,):
        if num_gates is None:
            num_gates = dims
        inputs = KL.Input(shape = (None, None, dims))
        x = KL.GlobalAveragePooling2D()(inputs)
        
        x = KL.Lambda(lambda f : tf.expand_dims(\
                                 tf.expand_dims(f, axis = 1), axis = 1))(x)
            
        x = KL.Conv2D(filters = dims // reduction,
                      kernel_size = 1,
                      use_bias = True,
                      name = name + '.fc1',
                      activation = 'relu',
                      padding = 'valid')(x)
        
        x = KL.Conv2D(filters = num_gates,
                      kernel_size = 1,
                      use_bias = True,
                      name = name + '.fc2',
                      activation = gate_activation,
                      padding = 'valid')(x)
        
        if not return_gates:
            x =  KL.Lambda(lambda f : f[0] * f[1])([inputs, x])
        
        return KM.Model(inputs, x)
        
     
    def OSBlockINin(self, x, filters, name, reduction = 4, T = 4, **kwargs):
        return self.OSBlock(x, filters, name, reduction, T, IN = True, bn = False, **kwargs)
     
    def OSBlock(self, x, filters, name, reduction = 4, T = 4, IN = False, bn = True, **kwargs):
        assert T >= 1
        assert filters >= reduction and filters % reduction == 0
        
        mid_channels = filters // reduction
        
        identity = x
        x1 = self.Conv1x1(x, filters = mid_channels, name = name + '.conv1')
        
        branches = []
        
        gate = self.ChannelGate(mid_channels, name = name + '.gate')
        
        for t in range(1, T + 1):
            x2 = self.LightConvStream(x1, 
                                      mid_channels, 
                                      t, 
                                      name = name + '.conv2.%d'%(t-1))
            x2 = gate(x2)
            branches.append(x2)
            
        x = KL.Add()(branches)
            
        x = self.Conv1x1Linear(x, filters, name = name + '.conv3', bn = bn)
        
        if IN:
            x = InstanceNormalization(epsilon = 1e-5,
                                   name = name + '.IN')(x)
        
        if int(identity.shape[-1]) != filters:
            identity = self.Conv1x1Linear(identity, 
                                          filters, 
                                          name = name + '.downsample')
        x = KL.Lambda(lambda x : x[0] + x[1])([x, identity])
        x = KL.Activation('relu')(x)
        return x
    
    def _make_layer(self, x, blocks, filters, name):
        for i in range(len(blocks)):
            x = blocks[i](x, filters, name = name + '.%d'%i)
        return x
        
    def _construct_fc_layer(self, x, fc_dims, input_dim, name, dropout_p = None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return x
        
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        
        for dim in fc_dims:
            x = KL.Dense(units = dim, name = name + '.0')(x)
            x = KL.BatchNormalization(epsilon = 1e-5,
                                      name = name + '.1')(x)
            x = KL.Activation('relu')(x)
            
        return x
    
    def load_weights(self, model,):
        torch_weights = torch.load('weights/osnet.pth')
        torch_weights_dict = {}
        for key in torch_weights.keys():
            weights = torch_weights[key]
            size = np.array(list(weights.shape)).size
            if size == 0:
                continue
            elif size == 1:
                torch_weights_dict[key] = weights.numpy()
            elif size == 2:
                torch_weights_dict[key] = weights.numpy().transpose([1, 0])
            elif size == 4:
                torch_weights_dict[key] = weights.numpy().transpose([2, 3, 1, 0])
            else:
                print(size, key)
                
                
        def parse_layer(_model):
            parsed_weights = []
            for layer in _model.layers:
                if hasattr(layer, 'layers'):
                    parsed_weights += parse_layer(layer)
                else:
                    weights = layer.weights
                    for weight in weights:
                        parsed_weights.append(weight)
            return parsed_weights
        
        keras_weights = parse_layer(model)
        
        keras_weights_dict = {}
        for weight in keras_weights:
            keras_weights_dict[weight.name] = weight
        
        weight_value_tuples = []
        for key in keras_weights_dict.keys():
            torch_key = key.replace('/', '.').\
                replace('depthwise_kernel:0', 'weight').\
                replace('kernel:0', 'weight').\
                replace('gamma:0', 'weight').\
                replace('bias:0', 'bias').\
                replace('beta:0', 'bias').\
                replace('moving_mean:0', 'running_mean').\
                replace('moving_variance:0', 'running_var')
            if torch_key not in torch_weights_dict.keys():
                print(key, torch_key)
                
            keras_weight = keras_weights_dict[key]
            torch_weight = torch_weights_dict[torch_key]
            if tuple(map(int, list(keras_weight.shape))) == torch_weight.shape:
                weight_value_tuples.append((keras_weights_dict[key], 
                                            torch_weights_dict[torch_key]))
            else:
                if 'depthwise_kernel:0' in key.split('/'):
                    weight_value_tuples.append((keras_weights_dict[key], 
                                            torch_weights_dict[torch_key].transpose([0, 1, 3, 2])))
                    
                else:
                    print(key, torch_key, keras_weight.shape, torch_weight.shape)
                
        K.batch_set_value(weight_value_tuples)
        
        model.save_weights('weights/osnet.h5')
        print('keras model weights saved to weights/osnet.h5')
            
    
    def graph(self, inputs = None):
        name = 'module'
        if inputs == None:
            inputs = KL.Input(shape = (None, None, 3), name = 'osnet_input_image')
        x = KL.ZeroPadding2D((3, 3), name = 'input_padding')(inputs)
        x = self.ConvLayer(x, 
                           self.conv1_IN,
                           filters = self.channels[0],
                           kernel_size = (7, 7),
                           strides = (2, 2),
                           use_bias = False,
                           padding = 'valid',
                           name = name + '.conv1')
        
        x = KL.ZeroPadding2D((1, 1))(x)
        x = KL.MaxPooling2D(pool_size = (3, 3),
                            strides = (2, 2),
                            padding = 'valid')(x)
        
        x = self._make_layer(x, self.blocks[0], self.channels[1], 
                             name = name + '.conv2')
        
        x = self.Conv1x1(x, self.channels[1], name = name + '.pool2.0')
        
        x = KL.AveragePooling2D(pool_size = 2,
                                strides = 2)(x)
        
        x = self._make_layer(x, self.blocks[1], self.channels[2],
                             name = name + '.conv3')
        
        x = self.Conv1x1(x, self.channels[2], name = name + '.pool3.0')
        x = KL.AveragePooling2D(pool_size = 2,
                                strides = 2)(x)
        
        x = self._make_layer(x, self.blocks[2], self.channels[3],
                             name = name + '.conv4')
        
        x = self.Conv1x1(x, self.channels[3], name = name + '.conv5')
        
        x = KL.GlobalAveragePooling2D()(x)
        
        x = self._construct_fc_layer(x, self.feature_dim, self.channels[3],
                                     name = name + '.fc')
        
        x = KL.Lambda(lambda f : tf.nn.l2_normalize(f, dim = -1), name = 'feats')(x)
        
        return KM.Model(inputs, x, name = 'keras_osnet')
        
if __name__ == '__main__':
    osnet = keras_osnet()
    model = osnet.graph()
    model.summary()
    osnet.load_weights(model)
