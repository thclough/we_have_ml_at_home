#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:38:26 2024

@author: Tighe_Clough
"""


# can create classes and pass attributes such as scaling factors

import numpy as np


def GlorotUniform(input_shape, output_shape, seed):
    
    rng = np.random.default_rng(seed)
    
    limit = (6 ** .5) / ((input_shape + output_shape) ** .5)
    
    weights = rng.uniform(low=-limit, high=limit, size=(input_shape, output_shape))
    
    return weights

def Orthogonal(input_shape, output_shape, seed):
    
    rng = np.random.default_rng(seed)
    
    normal_weights = rng.normal(size=(input_shape, output_shape))
    
    q,r = np.linalg.qr(normal_weights)
    
    return q

def RandomNormal(input_shape, output_shape, seed):
        
    rng = np.random.default_rng(seed)

    return rng.normal(size=(input_shape, output_shape)) * np.sqrt(1 / input_shape)

def He(input_shape, output_shape, seed):
    pass
    