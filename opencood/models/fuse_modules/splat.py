"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from einops.layers.torch import Rearrange, Reduce
import pdb

__all__ = ['SplAtConv2d', 'DropBlock2D']

class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)

            # pdb.set_trace()


            gap = sum(splited) 
        else:
            gap = x

        # print('----------------------------1 gap size', gap.shape)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        
        # pdb.set_trace()
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        ########--------------------------------------> V2X-VSS module


        ########--------------------------------------> V2X-VSS module
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            # pdb.set_trace()
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            # x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            # x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
    



# ##############################------------------------------------------------->   V2X-mamba
# from opencood.models.fuse_modules.mamba  import CoVSSBlock    
# class VSS_SplAtConv2d(Module):
#     """Split-Attention Conv2d
#     """
#     def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
#                  dilation=(1, 1), groups=1, bias=True,
#                  radix=2, reduction_factor=4,
#                  rectify=False, rectify_avg=False, norm_layer=None,
#                  dropblock_prob=0.0,
#                  vss_block=CoVSSBlock(in_chans=256,hidden_dim=256,patch_size=1), **kwargs):
#         super(VSS_SplAtConv2d, self).__init__()
#         padding = _pair(padding)
#         self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
#         self.rectify_avg = rectify_avg
#         inter_channels = max(in_channels*radix//reduction_factor, 32)
#         self.radix = radix
#         self.cardinality = groups
#         self.channels = channels
#         self.dropblock_prob = dropblock_prob
#         if self.rectify:
#             from rfconv import RFConv2d
#             self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
#                                  groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
#         else:
#             self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
#                                groups=groups*radix, bias=bias, **kwargs)
#         self.use_bn = norm_layer is not None
#         if self.use_bn:
#             self.bn0 = norm_layer(channels*radix)
#         self.relu = ReLU(inplace=True)
#         self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
#         if self.use_bn:
#             self.bn1 = norm_layer(inter_channels)
#         self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
#         if dropblock_prob > 0.0:
#             self.dropblock = DropBlock2D(dropblock_prob, 3)
#         self.rsoftmax = rSoftMax(radix, groups)



#         ########V2X-VSS
#         self.vss_block = vss_block
#         self.pre_vss = nn.Sequential(
#             Rearrange('(b m) c h w  -> b c m (h w)',m=5)  # bs*cavs, channel, H, W ----> bs,  channel, cavs, H*W
#         )
#         self.post_vss = nn.Sequential(
#             Rearrange('b m (h w) c -> (b m) c h w', w=1)  #bs, channel, cavs H*W ----> bs,  cavs, channel H*W
#         )

#         # self.fusion = nn.Sequential(
#         #     Rearrange('(b m) c h w -> b m c h w', m=5),
#         #     Reduce('b m c h w -> b c h w', 'mean')
#         #     )

#         self.fusion_mean = nn.Sequential(
#             Rearrange('(b m) c h w -> b m c h w', m=5),
#             Reduce('b m c h w -> b c h w', 'mean')
#             )
#         self.fusion_max = nn.Sequential(
#             Rearrange('(b m) c h w -> b m c h w', m=5),
#             Reduce('b m c h w -> b c h w', 'max')
#             )
            

#     def forward(self, x):
#         x = self.conv(x)
#         if self.use_bn:
#             x = self.bn0(x)
#         if self.dropblock_prob > 0.0:
#             x = self.dropblock(x)
#         x = self.relu(x)


#         batch, rchannel = x.shape[:2]
#         if self.radix > 1:
#             if torch.__version__ < '1.5':
#                 splited = torch.split(x, int(rchannel//self.radix), dim=1)
#             else:
#                 splited = torch.split(x, rchannel//self.radix, dim=1)


#             # pdb.set_trace()

            
#             gap = sum(splited) 
#         else:
#             gap = x

#         # print('----------------------------1 gap size', gap.shape)
#         gap = F.adaptive_avg_pool2d(gap, 1)
#         #################

#         gap = self.pre_vss(gap)
#         gap = self.vss_block(gap)
#         gap = self.post_vss(gap)

#         ################
#         # print('----------------------------2 gap size after pool', gap.shape)
#         gap = self.fc1(gap)

#         if self.use_bn:
#             gap = self.bn1(gap)
#         gap = self.relu(gap)
        
#         # print('----------------------------3 gap size', gap.shape)

#         atten = self.fc2(gap)
#         # print('----------------------------4 atten size', atten.shape)


#         atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
#         # print('----------------------------4 atten size', atten.shape)
#         ########--------------------------------------> V2X-VSS module
#         # atten = self.fusion(atten)
#         # pdb.set_trace()
#         # print('----------------------------5 atten size', atten.shape)

#         ########--------------------------------------> V2X-VSS module
#         if self.radix > 1:
#             if torch.__version__ < '1.5':
#                 attens = torch.split(atten, int(rchannel//self.radix), dim=1)
#             else:
#                 attens = torch.split(atten, rchannel//self.radix, dim=1)
#             # pdb.set_trace()
#             out = sum([att*split for (att, split) in zip(attens, splited)])
#         else:
#             out = atten * x

#         # print('----------------------------6 out size', out.shape)
#         # out = self.fusion(out)
            

#         out = self.fusion_max(out) + self.fusion_mean(out)

        
#         return out.contiguous()
    


# ##############################------------------------------------------------->   V2X-mamba
    
##############################------------------------------------------------->   V2X-mamba
from opencood.models.fuse_modules.mamba  import CoVSSBlock    
class VSS_SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0,
                 vss_block=CoVSSBlock(in_chans=256,hidden_dim=256,patch_size=1), **kwargs):
        super(VSS_SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)



        ########V2X-VSS
        self.vss_block = vss_block
        self.pre_vss = nn.Sequential(
            Rearrange('(b m) c h w  -> b c m (h w)',m=5)  # bs*cavs, channel, H, W ----> bs,  channel, cavs, H*W
        )
        self.post_vss = nn.Sequential(
            Rearrange('b m (h w) c -> (b m) c h w', w=1)  #bs, channel, cavs H*W ----> bs,  cavs, channel H*W
        )

        # self.fusion = nn.Sequential(
        #     Rearrange('(b m) c h w -> b m c h w', m=5),
        #     Reduce('b m c h w -> b c h w', 'mean')
        #     )

        self.fusion_mean = nn.Sequential(
            # Rearrange('(b m) c h w -> b m c h w', m=5),
            Reduce('b m c h w -> b c h w', 'mean')
            )
        self.fusion_max = nn.Sequential(
            # Rearrange('(b m) c h w -> b m c h w', m=5),
            Reduce('b m c h w -> b c h w', 'max')
            )
            
        #----------------------->
        self.pre_process = nn.Sequential(
            Rearrange('(b m) c h w -> b m c h w', m=5)
            )
        self.post_process = nn.Sequential(
            Rearrange('b m c h w -> (b m) c h w')
            )

    def forward(self, x):
        
        #--------->

        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        cavs = 5
        x = self.pre_process(x)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(1), dim=1)
            else:
                splited = torch.split(x, 1, dim=1)
          
            # gap = sum(splited) 
            # pdb.set_trace()
            gap = x
        else:
            gap = x

        # print('----------------------------1 gap size', gap.shape)
            
        gap = self.post_process(gap)
        
        gap = F.adaptive_avg_pool2d(gap, 1)
        #################


        gap = self.pre_vss(gap)
        gap = self.vss_block(gap)
        gap = self.post_vss(gap)

        ################
        # print('----------------------------2 gap size after pool', gap.shape)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        
        # print('----------------------------3 gap size', gap.shape)

        # atten = self.fc2(gap)


        
        # print('----------------------------4 atten size', atten.shape)
        atten = self.pre_process(gap)
        atten = F.softmax(atten, dim=1)


        # atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        # print('----------------------------4 atten size', atten.shape)
        ########--------------------------------------> V2X-VSS module
        # atten = self.fusion(atten)
        # pdb.set_trace()
        # print('----------------------------5 atten size', atten.shape)

        ########--------------------------------------> V2X-VSS module
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(1), dim=1)
            else:
                attens = torch.split(atten, 1, dim=1)
            # pdb.set_trace()
            out = [att*split for (att, split) in zip(attens, splited)]
        else:
            out = atten * x

        out = torch.cat(out,1)

        # print('----------------------------6 out size', out.shape)
        # out = self.fusion(out)
            

        out = self.fusion_max(out) + self.fusion_mean(out)

        
        return out.contiguous()
    


##############################------------------------------------------------->   V2X-mamba




if __name__ == "__main__":


    from opencood.models.fuse_modules.mamba import CoVSSBlock
    vss_block = CoVSSBlock(hidden_dim=256,patch_size=1).cuda()

    import torch
    # Define the parameters for the SplAtConv2d module
    in_channels = 256  # Number of input channels
    channels = 256    # Number of output channels
    kernel_size = 3  # Kernel size for the convolution
    stride = (1, 1)  # Stride for the convolution
    padding = (1, 1)  # Padding for the convolution
    dilation = (1, 1)  # Dilation for the convolution
    groups = 4        # Grouping for grouped convolution
    radix = 4        # Radix for split attention
    reduction_factor = 4  # Reduction factor for calculating inter_channels

    # Initialize the SplAtConv2d module
    # splat_conv2d = SplAtConv2d(in_channels, channels, kernel_size, stride, padding,
    #                            dilation, groups, bias=True, radix=radix, 
    #                            reduction_factor=reduction_factor, norm_layer=BatchNorm2d, vss_block=vss_block)

    splat_conv2d = VSS_SplAtConv2d(in_channels, channels, kernel_size, stride, padding,
                               dilation, groups, bias=True, radix=radix, 
                               reduction_factor=reduction_factor, norm_layer=BatchNorm2d, vss_block=vss_block).cuda()
    


    
    # Create a dummy input tensor of shape (batch_size, in_channels, height, width)
    # For this example, let's use a batch size of 1, and spatial dimensions of 64x64
    input_tensor = torch.randn(10, in_channels, 24, 88).cuda()
    print("Input tensor shape:", input_tensor.shape)

    
    # Forward pass through the SplAtConv2d module
    output_tensor = splat_conv2d(input_tensor)
    
    # Print the shape of the output tensor
    print("Output tensor shape:", output_tensor.shape)
