import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models

import pdb


def tv_loss(x, beta=0.5, reg_coeff=5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
    a, b, c, d = x.shape
    return reg_coeff * (torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta)) / (a * b * c * d))


class LossFunc(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_basic=10.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)

class WaveletLoss(nn.Module):
    def __init__(self):
        super(WaveletLoss, self).__init__()
        self.pooling = WaveletPool()
        self.charbonier = CharbonnierLoss()
    def forward(self,pred,gt):
        loss = 0
        pred_LL, pred_pool = self.pooling(pred)
        gt_LL, gt_pool = self.pooling(gt)
        loss += self.charbonier(pred_pool,gt_pool)
        pred_LL_2, pred_pool_2 = self.pooling(pred)
        gt_LL_2, gt_pool_2 = self.pooling(gt)
        loss += self.charbonier(pred_pool_2,gt_pool_2)
        _, pred_pool_3 = self.pooling(pred)
        _, gt_pool_3 = self.pooling(gt)
        loss += self.charbonier(pred_pool_3,gt_pool_3)
        return loss
class WaveletPool(nn.Module):
    def __init__(self, eps=1e-3):
        super(WaveletPool, self).__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL,torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss



def eval_noise_Loss(pred, ground_truth):
    
    # l1_loss = nn.L1Loss()
    # l2_loss = nn.MSELoss()
    smooth_l1_loss = nn.SmoothL1Loss()

    return smooth_l1_loss(pred, ground_truth)



class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        # pdb.set_trace()
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))

class LossAnneal_i(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal_i, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth[:,i,...])
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss

class LossAnneal(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        # loss = 0
        # for i in range(pred_i.size(1)):
        #     loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        # loss /= pred_i.size(1)
        loss = self.loss_func(pred_i, ground_truth)
        # loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )
class BasicLoss(nn.Module):
    def __init__(self, eps=1e-3, alpha=0.998, beta=100):
        super(BasicLoss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss(eps)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, burst_pred, gt, gamma):
        b,N,c,h,w = burst_pred.size()
        burst_pred = burst_pred.view(b,c*N,h,w)
        burst_gt = torch.cat([gt[..., i::2, j::2] for i in range(2) for j in range(2)], dim=1)

        anneal_coeff = max(self.alpha ** gamma * self.beta, 1)

        burst_loss = anneal_coeff * (self.charbonnier_loss(burst_pred, burst_gt))

        single_loss = self.charbonnier_loss(pred, gt)

        loss = burst_loss + single_loss

        return loss, single_loss, burst_loss
class AlginLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(AlginLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        y = F.pad(y,[1,1,1,1])
        diff0 = torch.abs(x-y[:,:,1:-1,1:-1])
        diff1 = torch.abs(x-y[:,:,0:-2,0:-2])
        diff2 = torch.abs(x-y[:,:,0:-2,1:-1])
        diff3 = torch.abs(x-y[:,:,0:-2,2:])
        diff4 = torch.abs(x-y[:,:,1:-1,0:-2])
        diff5 = torch.abs(x-y[:,:,1:-1,2:])
        diff6 = torch.abs(x-y[:,:,2:,0:-2])
        diff7 = torch.abs(x-y[:,:,2:,1:-1])
        diff8 = torch.abs(x-y[:,:,2:,2:])
        diff_cat = torch.stack([diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8])
        diff = torch.min(diff_cat,dim=0)[0]
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False,bn=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if bn :
            self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
                    nn.ReLU()
                )

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm




class Repaired_Net(nn.Module):
    def __init__(self,in_channel=3, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(Repaired_Net, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias

        self.color_channel = 3 if color else 1

        self.in_channel = in_channel
        self.out_channel = in_channel


        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # pdb.set_trace()

        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(self.in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, self.out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(self.out_channel, self.out_channel, 1, 1, 0)


        self.apply(self._init_weights)

        # self.weight_conv = nn.Sequential(
        #         nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1),
        #         nn.BatchNorm2d(self.in_channel,eps=1e-5, momentum=0.01, affine=True),
        #         nn.ReLU()
        #     )
        
        # self.pre_conv = nn.Sequential(
        #         nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1),
        #         nn.BatchNorm2d(self.in_channel,eps=1e-5, momentum=0.01, affine=True),
        #         nn.ReLU()
        #     )
        
        # self.data_conv = nn.Sequential(
        #         nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1),
        #         nn.BatchNorm2d(self.in_channel,eps=1e-5, momentum=0.01, affine=True),
        #         nn.ReLU()
        #     )
        

        # self.dropout = nn.Dropout(0.2)   

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)




    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """

        # self.confidence_aware =torch.ones_like(data_with_est)
        # data_with_est_1 = torch.where(data_with_est<1, data_with_est, self.confidence_aware)
        # data = torch.where(data<1, data, self.confidence_aware)

        ###############
        # data = self.dropout(data)
        # data_with_est = self.dropout(data_with_est)

        # self.confidence_aware_low =torch.zeros_like(data_with_est)
        # medium = torch.median(data)
        # data_with_est_1 = torch.where(data_with_est<medium, self.confidence_aware_low, data_with_est)
        # data = torch.where(data<medium, self.confidence_aware_low, data)



        # data_with_est = self.pre_conv(data_with_est)
        # data = self.data_conv(data)


        # data_with_est = self.dropout(data_with_est)################  1
        # data = self.dropout(data)################  1



        # self.confidence_aware_low =torch.zeros_like(data_with_est)
        # data_with_est_1 = torch.where(data_with_est<0, self.confidence_aware_low, data_with_est)
        # data = torch.where(data<0, self.confidence_aware_low, data)

        # self.confidence_aware_high =torch.ones_like(data_with_est)
        # data_with_est_1 = torch.where(data_with_est>1, self.confidence_aware_high, data_with_est)
        # data = torch.where(data>1, self.confidence_aware_high, data)


        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5,size=conv4.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6,size=conv3.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7,size=conv2.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        core = self.outc(F.interpolate(conv8, size=data.size()[-2:], mode=self.upMode, align_corners=True))


        self.confidence_aware_low =torch.zeros_like(data_with_est)
        core = torch.where(data_with_est<0, self.confidence_aware_low, core)
        data = torch.where(data<0, self.confidence_aware_low, data)

        self.confidence_aware_high =torch.ones_like(data_with_est)
        core = torch.where(core>1, self.confidence_aware_high, core)
        data = torch.where(data>1, self.confidence_aware_high, data)

        # core = self.dropout(core)################  1

        # return self.weight_conv(data + core)
        return data + core



