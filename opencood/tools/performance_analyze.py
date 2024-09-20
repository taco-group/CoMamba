# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt


##########
from opencood.models.fuse_modules.estimation_time import analyze
import torch
import time
import pdb

# torch.backends.cudnn.enabled=True


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    opt = parser.parse_args()
    return opt



def main():

    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    # hypes = yaml_utils.load_yaml(None, opt)
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # print('Dataset Building')
    # opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # print(f"{len(opencood_dataset)} samples found.")
    # data_loader = DataLoader(opencood_dataset,
    #                          batch_size=1,
    #                          num_workers=1,
    #                          collate_fn=opencood_dataset.collate_batch_test,
    #                          shuffle=False,
    #                          pin_memory=False,
    #                          drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    torch.cuda.empty_cache()
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    # saved_path = opt.model_dir
    # _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()


    cavs_num = 15

    point = 29012
    #------------------> dataset
    batch_data = {}
    batch_data['ego'] = {}
    batch_data['ego']['processed_lidar'] = {}
    batch_data['ego']['processed_lidar']['voxel_features'] = torch.rand(point, 32, 4, dtype=torch.float)
    batch_data['ego']['processed_lidar']['voxel_coords'] = torch.rand(point, 4, dtype=torch.float)*cavs_num
    batch_data['ego']['processed_lidar']['voxel_num_points'] = torch.rand(point, dtype=torch.float)
    batch_data['ego']['record_len'] = torch.tensor([cavs_num])
    batch_data['ego']['spatial_correction_matrix']=torch.rand(1, 5, 4, 4, dtype=torch.float)
    # batch_data['ego']['prior_encoding'] = torch.zeros(1, 5, 3)
    batch_data['ego']['prior_encoding'] = torch.zeros(1, cavs_num, 3)
    # batch_data['ego']['prior_encoding'][:, :, 0] = cavs_num * torch.rand(1, 5) - 1
    batch_data['ego']['prior_encoding'][:, :, 0] = cavs_num * torch.rand(1, cavs_num) - 1


    #-------------------------->
    import time
    num_runs = 50
    print('the total number is ', num_runs)
    start_time = time.time()
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs)):

            torch.cuda.synchronize()
            # print(i)
            

            batch_data = train_utils.to_device(batch_data, device)
            cav_content = batch_data['ego']
            output = model(cav_content)
            
    #--------------------------->
    end_time = time.time()

    total_time = end_time - start_time
    analyze(model,batch_data['ego'], num_runs,total_time)



if __name__ == '__main__':
    main()
