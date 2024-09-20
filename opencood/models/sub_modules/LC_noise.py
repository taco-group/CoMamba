'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-05-04 10:50:25
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2024-01-22 10:07:24
'''

# import sys
# sys.path.append('/home/jinlong/4.3D_detection/Noise_V2V/v2vreal')
# sys.path.append('/home/jinlong/4.3D_detection/Noise_V2V/v2vreal/opencood')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
import os
import pdb


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True




def data_dropout_each_agent(x, p,key_max):

    #####
    ##### add the noise to each agent of the feature map of each  channel  by random.
    # x: [N, C, W, H]
    # p: probability of dropout
    # key_max: the maximum number of noise
    #####
    setup_seed(100)

    if p <0. or p >=1:

        raise ValueError('p must be in interval[0, 1]')

    size = x.size()
    N,C, W, H = size
    # print(N,C, W, H)
    
    list_channel = random.sample(range(0, C), int(p*C))
    for w in range(N-1):
        for i in range(len(list_channel)):
            random_tensor =  torch.from_numpy(np.random.uniform(low=0, high=key_max, size =(1,1,W,H)))
            x[w+1:w+2,list_channel[i],:,:] = random_tensor.clone().cuda()
        
        return x
    


def data_dropout_feature_with_channel(feature, p,key_max, record_len):
    setup_seed(100)
    
    #####
    # x: [N, C, W, H]
    # p: probability of dropout
    # key_max: the maximum number of noise
    # record_len: the number of group of agents
    #####

    if p <0. or p >=1:

        raise ValueError('p must be in interval[0, 1]')
        


    split_x = []
    split_xx = regroup(feature, record_len)
    for xx in split_xx:
        if xx.size()[0] > 1:# generating training noise except ego feature map
            x = data_dropout_each_agent(xx.clone(), p=p, key_max=key_max)
            split_x.append(x)
        else:
            split_x.append(xx)
    features_2d = torch.cat(split_x, dim=0)

    return features_2d

        
    
    
def data_dropout_uniform(x, p, key_max):
    setup_seed(100)


    if p <0. or p >=1:

        raise ValueError('p must be in interval[0, 1]')

    one_prob = 1. - p

    xx=x.clone()

    size = x[1:,].size()
    random_tensor = np.random.binomial(n=1, p = one_prob, size = size)

    random_tensor = torch.from_numpy(random_tensor)

    # print("random_tensor", random_tensor)
    adverse_random_tensor = random_tensor.clone()

    one_tensor =  torch.ones(size)

    adverse_random_tensor = one_tensor-adverse_random_tensor


    noise=torch.randint(0, int((key_max)*100), size)
    noise=noise/100

    # print("before_noise: ", noise)

    noise = noise*adverse_random_tensor

    # print("after_noise: ", noise)

    x[1:,] = x[1:,]*random_tensor.cuda()
    x[1:,] = x[1:,]+noise.cuda()

    # print("Noise percentage#########################: ", torch.all(torch.eq(xx,x)))
    return  x







def lighting_denoising(input_tensor, num_iterations=2):
    """
    Replace values in the input Tensor with the median value, repeating the process for a specified number of iterations.

    Args:
    - input_tensor: Input PyTorch Tensor.
    - num_iterations: Number of iterations to repeat the replacement process, default is 2.

    Returns:
    - New Tensor after replacement.
    """
    # Flatten the input Tensor to a 1D array
    flattened_tensor = input_tensor.view(-1)

    for _ in range(num_iterations):
        # Calculate the median value
        median_value = torch.median(flattened_tensor)

        # Replace values greater than the median with the median
        flattened_tensor[flattened_tensor > median_value] = median_value

    # Reshape the flattened array back to the original Tensor shape
    output_tensor = flattened_tensor.view(input_tensor.shape)

    return output_tensor


if __name__== '__main__':

    # input = torch.linspace(4, 6, steps=900).view(30, 30).cuda()

    input = torch.ones(2,3,2,5).cuda()
    input = input*100
    print(input)
    # output = data_dropout(input, p=0.5)
    output=data_dropout_uniform(input, p=0.5,key_max=10)
    print(output)


    # input = torch.ones(2,4,2,5).cuda()
    # print(input)
    # output = data_dropout_each_agent(input, p=0.99, key_max=10)
    # print(output)




    






