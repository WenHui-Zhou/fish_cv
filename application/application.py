#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-10-06
# Copyright (C)2020 All rights reserved.

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import tensorflow as tf

import abc

class Application:
    '''
    the base class for all applications. An application is defined by his dataset,
    inputs(data provider and shuffling),model and loss.An applcation object is used
    to build the complete computation graph for training or evaluation.
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self,hparams,data_path,mode = tf.estimator.ModeKeys.TRAIN):
        '''
        hparams: a tf.contrib.training.Hparams objects,holding hyperparameters used to
        build and train a model,such as the number of hidden units in a neural net
        layer,or the learning rate to use when training.
        data-path:a path name to the data.
        mode: the mode used by the tf estimator, train or eval
        '''
        self.hparams = hparams
        self.data_path = data_path
        
        self.dataset = None
        self.model = None
        self.total_loss = None
        self._mode = mode

    @abc.abstractmethod
    def get_dataset(self,data_path):
        '''
        create and return a TF slim Dataset object wrapping the data in the 
        data_path,this object will be used as the parameter of 'get_inputs'
        '''
        pass

    @abc.abstractmethod
    def get_inputs(self,dataset):
        '''
        create and return an instance of a tuple class 'inputEndpoints' which define
        with collections.namedtuple, the fields within this tuple are application
        -specific tensors,sub-(computaion-)graph for raw data parsing,transformations
        reading,and shuffling is built here(using data provider,dataset,shuffle)
        '''
        pass

    @abc.abstractmethod
    def get_model(self,input_data):
        '''
        create and return an instance of model.net.Net,whose 'build()' method is used
        to build the sub-(computation-)graph corresponding to the model/net (including loss)
        link it to the sub(-computaion-)graph built with 'get_inputs' and return the
        tensor of loss.
        parameters:
            input_data    an instance of a tuple class 'InputEndpoints' which holds 
            named input tensors to the model-graph.
        returns:
            an instance of model.net.net whose 'build()' method is used to build a
            sub-graph of the model and return the tensor of loss.
        '''
        pass

    def get_loss(self):
        '''
        get the tensor corresponding to the loss. 'build' the model if necessary
        '''
        if self.model is None:
            raise ValueError('Model has not been built')
        if self.total_loss is None:
            self.total_loss = self.model.build()
            tf.losses.add_loss(self.total_loss)
        return self.total_loss

    def get_train_step_fn(self):
        '''
        recall function when finishing the training step.
        '''
        return None

    def get_tensor_list(self):
        '''
        get predict and label
        '''
        return None

    def setup(self,is_training=True):
        """
        build the complete computation-graph from raw data input to model output and 
        loss. the dataset,the inputs to the model,the model,and the loss are kept as
        member fields as this object.
        """
        global g_input_data
        self.is_training = is_training

        if g_input_data is None:
            with tf.device('/cpu:0'):
                dataset = self.get_dataset(self.data_path)
                input_data = self.get_inputs(dataset)
                g_input_data = input_data
        model = self.get_model(g_input_data)
        self.total_loss = model.build()
        tf.losses.add_loss(self.total_loss)

if __name__ == '__main__':
    print('run')
