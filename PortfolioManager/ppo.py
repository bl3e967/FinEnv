import os, sys
import tensorflow as tf 
import numpy as np
import FinEnv
import multiprocessing as mp 
from warnings import warn

'''
Objective function is the reward function: 1/tf*sum( r(t) )
'''

class PPO(): 
    def __init__(self, m, n, f, network_type='CNN'):
        '''
        Args: 
            network_type: string. 'CNN', 'RNN', 'LSTM'
        '''
        # tensorflow session
        self.sess = tf.Session()

        self.M = m; self.N = n; self.F = f 
        
        self.network_type_dict = {
                'CNN' : self._CNN, 
                'LSTM': self._LSTM, 
                'RNN' : self._RNN, 
            }
        self.network = self.network_type_dict[network_type]
    
    def train(self): 
        # TODO: Not complete
        # input shape = [batch, in_height, in_width, in_channels]
        inputTensor = tf.placeholder(tf.float32, [None, self.M, self.N, self.F], 'InputPriceTensor')
        pvm_input = tf.placeholder(tf.float32, [20, 11, 1, 1], 'PVMInput')
        output = self.network(inputTensor, pvm_input)
        raise Warning("Not implemented")

    def _CNN(self, inputTensor, pvm_input):     
        # filter = [filter_height, filter_width, in_channels, out_channels]  
        # (3x11x50) Input 
      
        # l1: shape (2x11x48)
        self.l1 = tf.nn.conv2d(inputTensor, filter=[1,3,3,2], activation_fn=tf.nn.relu)
        # l2: shape (1x11x1)
        self.l2 = tf.nn.conv2d(self.l1, filter=[1,48,2,1], activation_fn=tf.nn.relu)
        # l3: shape (11x1)
        self.l3 = tf.squeeze(self.l2, axis=0)
        # l4: shape (20+1, 11, 1)
        self.l4 = tf.concat([pvm_input, self.l3], axis=0)
        # l5: shape (11, 1)
        self.l5 = tf.nn.conv2d(self.l4, filter=[1,1,21,1], activation_fn=tf.nn.relu)

        return self.l5 

    def _LSTM(self, inputTensor, pvm_input): 
        raise Warning('LSTM not implemented yet')
    
    def _RNN(self, inputTensor, pvm_input): 
        raise Warning('RNN not implemented yet')

    def _check_CNN_outputs(self, feed_dict): 
        def _run_check(vector, shape): 
            if shape != np.shape(vector): 
                raise Exception('shape of vector {}, not {}'.format(np.shape(vector), shape))
        
        output_list = [self.l1, self.l2, self.l3, self.l4, self.l5]
        l1, l2, l3, l4, l5 = self.sess.run(output_list, feed_dict)

        _run_check(l1, [2,11,48])
        _run_check(l2, [1,11,1])
        _run_check(l3, [11,1])
        _run_check(l4, [21,11,1])
        _run_check(l5, [1,1,21,1])
        
    

class Worker(): 
    def __init__(self, wid): 
        '''
        Args: 
            wid: Worker id, string
        '''
        pass 
    
    def work(self): 
        pass

class Chief(): 
    def __init__(self): 
        pass 
    



