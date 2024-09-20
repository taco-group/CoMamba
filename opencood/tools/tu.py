import math

import math
import torch.nn.functional as F

from opencood.models.sub_modules.base_transformer import *
from opencood.models.fuse_modules.hmsa import *
from opencood.models.fuse_modules.mswin import *
from opencood.models.fuse_modules.v2xvit_basic import *
from opencood.models.fuse_modules.cswin import LePEAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from pytorch_benchmark import benchmark

from vit_pytorch.vit import Attention
from vit_pytorch.max_vit import Attention as Max_Attention
from axial_attention import AxialAttention

from torch.profiler import profile, record_function, ProfilerActivity



def _eval(model, inputs):
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        # Hardware warm-up
        for _ in range(20):
            inputs = inputs.to(device)
            # mask = mask.to(device)
            # spatial_correction_matrix = spatial_correction_matrix.to(device)
            # prior_encoding = prior_encoding.to(device)
            torch.cuda.synchronize()
            tm = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()
            _ = time.time() - tm

        # Main Evaluation
        elapsed = 0
        for iter_idx in range(50):

            inputs = inputs.to(device)
            torch.cuda.synchronize()
            tm = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()
            elapsed += time.time() - tm

        print(f"Average time: {elapsed / 50}")


if __name__ == "__main__":
    #### Testing with PoolingAttention
    # TEst
    # batch_size, num_cav, height, wight, channels
    # b l c h w
    input = torch.rand(2, 4, 128, 128, 256)
    prior_encoding = torch.ones(2, 4, 1, 1, 3)
    # mask bl h w c
    mask = torch.ones(2, 4)
    # B l 4 4
    padding_eye = np.tile(np.eye(4)[None], (4, 1, 1))
    spatial_correction_matrix = torch.from_numpy(padding_eye).unsqueeze(0).repeat(2, 1, 1, 1)
    args = {
        # number of fusion blocks per encoder layer
        'num_blocks': 1,
        # number of encoder layers
        'depth': 2,
        'num_scales': 3,
        'use_roi_mask': True,
        'use_RTE': True,
        'RTE_ratio': 2,  # 2 means the dt has 100ms interval while 1 means 50 ms interval
        # agent-wise attention
        'cav_att_config': {
            'dim': [256, 512, 1024],
            'use_hetero': True,
            'use_RTE': True,
            'RTE_ratio': 2,
            'heads': [8, 16, 32],
            'dim_head': [32, 32, 32],
            'dropout': 0.3,
        },
        # spatial-wise attention
        'pwindow_att_config': {
            'dim': [256, 512, 1024],
            'heads': [[16, 8, 4], [32, 16, 8], [64, 32, 16]],
            'dim_head': [[16, 32, 64], [16, 32, 64], [16, 32, 64]],
            'dropout': 0.3,
            'window_size': [4, 4, 4],
            'relative_pos_embedding': True,
            'fusion_method': 'split_attn',
            'pooling_stride': [1, 2, 4],
        },
        # feedforward condition,
        'feed_forward': {
            'mlp_dim': [256, 512, 1024],
            'dropout': 0.3,
        },
        'sttf': {
            'voxel_size': [0.4, 0.4, 4],
            'downsample_rate': 4,
        }
    }
    # net = V2XTEncoder(args)
    #
    # with profile(activities=[ProfilerActivity.CUDA],
    #              profile_memory=True, record_shapes=True) as prof:
    #     output = net(input, mask, spatial_correction_matrix, prior_encoding)
    #     print(f"output: {output.shape}")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    #
    # # benchmarking
    # _eval(net, input, mask, spatial_correction_matrix, prior_encoding)  # 0.479 seconds

    # benchmarking ViT
    # model = Attention(
    #     dim=256, heads=8, dim_head=32, dropout=0.
    # )
    # input = torch.rand(1, 128*128, 256)   # 0.01 --> 0.05 --> 0.25 s
    # # with profile(activities=[ProfilerActivity.CUDA],
    # #              profile_memory=True, record_shapes=True) as prof:
    # #     output = model(input)
    # #     print(f"output: {output.shape}")
    # # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))    # 4.09GB
    # _eval(model, inputs=input)

    ## Benchamrking MaxViT
    # model = Max_Attention(
    #     dim=256, dim_head=32, window_size=8
    # )
    # input = torch.rand(1, 16, 16, 8, 8, 256)
    # # with profile(activities=[ProfilerActivity.CUDA],
    # #              profile_memory=True, record_shapes=True) as prof:
    # #     output = model(input)
    # #     print(f"output: {output.shape}")
    # # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))    # 128.13 Mb
    # _eval(model, inputs=input)  # 0.002 seconds

    ## Axial Attention
    # model = AxialAttention(
    #     dim=256, dim_index=1, dim_heads=32, heads=8
    # )
    # input = torch.rand(1, 256, 128, 128)
    #
    # with profile(activities=[ProfilerActivity.CUDA],
    #                  profile_memory=True, record_shapes=True) as prof:
    #     output = model(input)
    #     print(f"output: {output.shape}")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))    # 304 Mb
    # _eval(model, inputs=input)  # 0.005 seconds

    # CSwin ATtetnion
    # model = LePEAttention(
    #     dim=256, resolution=128, idx=0, split_size=8, num_heads=8
    # )
    # input = torch.rand(1, 128*128, 256)
    #
    # with profile(activities=[ProfilerActivity.CUDA],
    #                  profile_memory=True, record_shapes=True) as prof:
    #     output = model(input)
    #     print(f"output: {output.shape}")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))    # 592 Mb
    # _eval(model, inputs=input)  # 0.008 seconds

    # MSwin attention
    # model = PyramidWindowAttention(
    #     dim=256,
    #     heads=[16, 8, 4],
    #     dim_heads=[16, 32, 64],
    #     drop_out=0.0,
    #     window_size=[4, 8, 16],
    #     relative_pos_embedding=True,
    #     fuse_method='split_attn',
    #     pooling_stride=None,
    # )
    # input = torch.rand(1, 1, 128, 128, 256)
    # with profile(activities=[ProfilerActivity.CUDA],
    #                      profile_memory=True, record_shapes=True) as prof:
    #         output = model(input)
    #         print(f"output: {output.shape}")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))    # 384 Mb
    # _eval(model, inputs=input)  # 0.008 seconds

    # MSPA
    model = PyramidWindowAttention(
        dim=256,
        heads=[16, 8, 4],
        dim_heads=[16, 32, 64],
        drop_out=0.0,
        window_size=[4, 4, 4],
        relative_pos_embedding=True,
        fuse_method='split_attn',
        pooling_stride=[1, 2, 4],
    )
    input = torch.rand(1, 1, 128, 128, 256)
    with profile(activities=[ProfilerActivity.CUDA],
                         profile_memory=True, record_shapes=True) as prof:
            output = model(input)
            print(f"output: {output.shape}")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))    # 245 Mb
    _eval(model, inputs=input)  # 0.006 seconds