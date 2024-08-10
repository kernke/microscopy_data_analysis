# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:53:31 2024

@author: kernke
"""

import numpy as np
import pytest


def pytest_configure():
    pytest.pattern = np.array([[5,8,3],[6,9,8]])
    pytest.shape=np.array([32,35])
    pytest.shift=np.array([4,20])

    a=np.zeros(pytest.shape)    
    a[pytest.pattern[0],pytest.pattern[1]]=1
    pytest.a=a
    
    b=np.zeros(pytest.shape)
    shifted=(pytest.pattern.T+pytest.shift).T
    b[shifted[0],shifted[1]]=1    
    pytest.b=b 
    
    c=np.zeros([2,*pytest.shape])
    c[0]=a
    c[1]=b
    pytest.stack=c
    
    
@pytest.fixture()
def central_pattern_img():
#    a=np.zeros(pytest.shape)    
#    a[pytest.pattern[0],pytest.pattern[1]]=1
    return pytest.a

@pytest.fixture()
def shifted_pattern_img():
#    b=np.zeros(pytest.shape)
#    shifted=(pytest.pattern.T+pytest.shift).T
#    b[shifted[0],shifted[1]]=1
    return pytest.b

@pytest.fixture()
def expected_shift():
    exp=np.zeros(2,dtype=int)
    for i in range(2):
        if pytest.shift[i]<=0:
            exp[i]=-pytest.shift[i]
        else:
            exp[i]=pytest.shape[i]-pytest.shift[i]
    return exp

@pytest.fixture()
def stack():
    return pytest.stack