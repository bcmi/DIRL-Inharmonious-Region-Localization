import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import scipy.stats as st
import numpy as np
from torch.nn.parameter import Parameter
from .blocks import Conv2dBlock, BasicBlock, BasicConv
import cv2
import copy

## ---------------------Bi-directional Feature Integration -----------------
class BidirectionFeatureIntegration(nn.Module):
    def __init__(self, in_ch_list, out_ch=64, fusion_mode='h2l'):
        super(BidirectionFeatureIntegration, self).__init__()
        self.n_input = len(in_ch_list)
        assert self.n_input > 0
        self.fusion_mode = fusion_mode
        self.downsample = nn.AvgPool2d(3,2,1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(True)
        if self.fusion_mode == 'h2l' or self.fusion_mode == 'l2h':
            l_in_ch = in_ch_list[0]
            h_in_ch = in_ch_list[1]

            self.top_down = Conv2dBlock(h_in_ch, l_in_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.bottom_up = Conv2dBlock(l_in_ch, h_in_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
            if self.fusion_mode == 'h2l':
                in_ch_ratio = 2
                self.h_concat = Conv2dBlock(h_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
                self.l_concat = Conv2dBlock(l_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            elif self.fusion_mode == 'l2h':
                in_ch_ratio = 2
                self.l_concat = Conv2dBlock(l_in_ch*in_ch_ratio, out_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
                self.h_concat = Conv2dBlock(h_in_ch*in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
                
        elif self.fusion_mode == 'hl2m' or self.fusion_mode == 'lh2m':
            l_in_ch = in_ch_list[0]
            m_in_ch = in_ch_list[1]
            h_in_ch = in_ch_list[2]
            
            self.top_down_h2m = Conv2dBlock(h_in_ch, m_in_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.top_down_m2l = Conv2dBlock(m_in_ch, l_in_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.bottom_up_m2h = Conv2dBlock(m_in_ch, h_in_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
            self.bottom_up_l2m = Conv2dBlock(l_in_ch, m_in_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)

            in_ch_ratio = 2
            self.l_concat = Conv2dBlock(l_in_ch * in_ch_ratio, out_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
            self.m_concat = Conv2dBlock(m_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.h_concat = Conv2dBlock(h_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
        else:
            raise NameError("Unknown mode:\t{}".format(fusion_mode))

    def forward(self, xl, xm=None, xh=None):
        if self.fusion_mode == 'h2l' or self.fusion_mode == 'l2h':
            # Bottom        xl ----> xh            Up
            #               |         \ 
            # Down           \   xl <----  xh      Top
            #                 \  /     \  /
            #                   C -> + <-C
            #                        ↓
            #                       out
            top_down_results = [xh]
            xh2l = self.top_down(F.interpolate(xh, scale_factor=2))
            top_down_results.insert(0, xl + xh2l)

            bottom_up_results = [xl]
            xl2h = self.bottom_up(xl)
            bottom_up_results.append(xh+xl2h)

            xl_cat = torch.cat([top_down_results[0],bottom_up_results[0]], dim=1)
            xh_cat = torch.cat([top_down_results[1],bottom_up_results[1]], dim=1)
            if self.fusion_mode == 'h2l':
                xh_cat = self.h_concat(F.interpolate(xh_cat, scale_factor=2))
                xl_cat = self.l_concat(xl_cat)
                
            elif self.fusion_mode == 'l2h':
                xh_cat = self.h_concat(xh_cat)
                xl_cat = self.l_concat(xl_cat)
            
            xout = xh_cat + xl_cat
            
        elif self.fusion_mode == 'hl2m' or self.fusion_mode== 'lh2m':
            # Bottom       xl ---->  xm ----> xh            Up
            #               \         \        \
            # Down           \   xl <----  xm <---- xh      Top
            #                 \  /      \  /     \  / 
            #                   C  ---->  C  <---- C
            #                             ↓
            #                            out
            top_down_results = [xh] 
            xh2m = self.top_down_h2m(F.interpolate(xh, scale_factor=2)) + xm
            top_down_results.insert(0, xh2m)
            xm2l = self.top_down_m2l(F.interpolate(xh2m, scale_factor=2)) + xl
            top_down_results.insert(0, xm2l)

            bottom_up_results = [xl]
            xl2m = self.bottom_up_l2m(xl) + xm
            bottom_up_results.append(xl2m)
            xm2h = self.bottom_up_m2h(xl2m) + xh
            bottom_up_results.append(xm2h)

            xl_cat = torch.cat([top_down_results[0],bottom_up_results[0]], dim=1)
            xm_cat = torch.cat([top_down_results[1],bottom_up_results[1]], dim=1)
            xh_cat = torch.cat([top_down_results[2],bottom_up_results[2]], dim=1)
            
            xl_cat = self.l_concat(xl_cat)
            xm_cat = self.m_concat(xm_cat)
            xh_cat = self.h_concat(F.interpolate(xh_cat, scale_factor=2))

            xout = xl_cat + xm_cat + xh_cat
        return xout

class Transition(nn.Module):
    def __init__(self, in_ch_list, out_ch_list):
        super(Transition, self).__init__()
        inch0, inch1, inch2, inch3, inch4 = in_ch_list
        outch0, outch1, outch2, outch3, outch4 = out_ch_list
        
        self.im0 = BidirectionFeatureIntegration([inch0,inch1], outch0, fusion_mode='h2l')
        self.im1 = BidirectionFeatureIntegration([inch0,inch1, inch2], outch1, fusion_mode='hl2m')
        self.im2 = BidirectionFeatureIntegration([inch1,inch2, inch3], outch2, fusion_mode='hl2m')
        self.im3 = BidirectionFeatureIntegration([inch2,inch3, inch4], outch3, fusion_mode='hl2m')
        self.im4 = BidirectionFeatureIntegration([inch3,inch4], outch4, fusion_mode='l2h')

    def forward(self, xs, gc=None):
        out_xs = []
        out_xs.append(self.im0(xl=xs[0], xh=xs[1]))
        out_xs.append(self.im1(xl=xs[0], xm=xs[1], xh=xs[2]))
        out_xs.append(self.im2(xl=xs[1], xm=xs[2], xh=xs[3]))
        out_xs.append(self.im3(xl=xs[2], xm=xs[3], xh=xs[4]))
        out_xs.append(self.im4(xl=xs[3], xh=xs[4]))
        return out_xs



## -----------------Mask-guided Dual Attention ---------------------


class ECABlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

## SA
def _get_kernel(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    """
        normalization
    :param in_:
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)

##

class SpatialGate(nn.Module):
    def __init__(self, in_dim=2, mask_mode='mask'):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.mask_mode = mask_mode
        
        self.spatial = nn.Sequential(*[
            BasicConv(in_dim, in_dim, 3, 1, 1),
            BasicConv(in_dim, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2,  relu=False)
        ])
        
        if 'gb' in mask_mode.split('_')[-1]:
            print("Using Gaussian Filter in mda!")
            gaussian_kernel = np.float32(_get_kernel(31, 4))
            gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
            self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, x):
        x_compress = x
        x_out = self.spatial(x_compress)
        attention = F.sigmoid(x_out) # broadcasting
        x = x * attention
        if 'gb' in self.mask_mode:
            soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
            soft_attention = min_max_norm(soft_attention)       # normalization
            x = torch.mul(x, soft_attention.max(attention))     # x * max(soft, hard)
        return x, attention
        
class MaskguidedDualAttention(nn.Module):
    def __init__(self, gate_channels, mask_mode='mask'):
        super(MaskguidedDualAttention, self).__init__()
        self.ChannelGate = ECABlock(gate_channels)
        self.SpatialGate = SpatialGate(gate_channels, mask_mode=mask_mode)
        self.mask_mode = mask_mode
    def forward(self, x):
        x_ca = self.ChannelGate(x)
        x_out, mask = self.SpatialGate(x_ca)
        return x_out + x_ca, mask

## -----------------Global-context Guided Decoder ---------------------
class GGDBlock(nn.Module):
    def __init__(self, channel=32, is_outmost=False):
        super(GGDBlock, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_inup = Conv2dBlock(channel, channel, 3, 1, padding=1, norm='bn', activation='relu', use_bias=False)
        self.conv_inbottom = Conv2dBlock(channel, channel, 3, 1, padding=1, norm='bn', activation='relu', use_bias=False)
        self.conv_cat = Conv2dBlock(channel*2, channel, 3, 1, padding=1, norm='bn', activation='relu', use_bias=False)

        self.outmost = is_outmost
        if self.outmost:
            self.conv4 = Conv2dBlock(channel, channel, 3, 1, padding=1, norm='bn', activation='relu', use_bias=False)
            self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x, up,bottom):
        #         x
        #         ↓
        # <-[C]-- * ---- Up
        # <--↑---------- Bottom 
        x_up = self.conv_inup(self.upsample(up)) * x # 28
        x_bottom = self.conv_inbottom(self.upsample(bottom)) # 56

        x_cat = torch.cat((x_up, x_bottom), 1) # 28
        x_out = self.conv_cat(x_cat) # 28

        xup_out = x_out
        xbottom_out = x_bottom
        
        if self.outmost:
            x_out = self.upsample(x_out)
            x = self.conv4(x_out)
            x = self.conv5(x_out)
            return {'xup':x, 'xbottom':x_out}
        else:
            return {'xup':xup_out, 'xbottom':xbottom_out}

class GGD(nn.Module):
    def __init__(self, channel=32, nstage=4):
        super(GGD, self).__init__()
        self.decoder = nn.ModuleDict()
        self.nstage = nstage - 1
        for i in range(self.nstage):
            if i == 0: self.decoder['d0'] = GGDBlock(channel=channel,  is_outmost=True)
            else:
                self.decoder['d{}'.format(i)] = GGDBlock(channel=channel, is_outmost=False)
    
    def forward(self, xs):
        #x0,x1,x2,x3,x4,x5=xs
        xup = xdown = xs[-1]
        for i, x in enumerate(
            xs[1:-1][::-1]
        ):
            idx = self.nstage - i - 1
            xout = self.decoder['d{}'.format(idx)](x, xup,xdown)
            xup,xdown = xout['xup'], xout['xbottom']
        return xup

## ----------------DIRL --------------------------------

class InharmoniousEncoder(nn.Module):
    def __init__(self, opt, n_channels=3):
        super(InharmoniousEncoder, self).__init__()
        if opt.backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.in_dims = [64, 128, 256, 512, 512]
        elif opt.backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.in_dims = [64, 256, 512, 1024, 2048]
        ## -------------Encoder--------------
        self.inconv = nn.Conv2d(n_channels,64,3,1,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True) #224,64
        self.maxpool = nn.MaxPool2d(3,2,1)
        #stage 1
        self.encoder1 = resnet.layer1 #112,64*4
        #stage 2
        self.encoder2 = resnet.layer2 #56,128*4
        #stage 3
        self.encoder3 = resnet.layer3 #28,256*4
        #stage 4
        self.encoder4 = resnet.layer4 #14,512*4
        self.encoder5 = nn.Sequential(*[
            BasicBlock(resnet.inplanes, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        ])
        
    def forward(self, x, backbone_features=None):
        hx = x
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)
        
        h1 = self.encoder1(hx) # 224
        h2 = self.encoder2(h1) # 112
        h3 = self.encoder3(h2) # 56
        h4 = self.encoder4(h3) # 28
        hx = self.maxpool(h4)
        h5 = self.encoder5(hx) # 14
        return {"skips":[h1,h2,h3,h4,h5]}

class InharmoniousDecoder(nn.Module):
    def __init__(self,opt, n_channels=3):
        super(InharmoniousDecoder,self).__init__()
        ## -------------Dimention--------------
        self.opt = opt
        if opt.backbone == 'resnet34':
            self.dims = [512,512,256,128,64,64]
        elif opt.backbone == 'resnet50':
            self.dims = [2048, 1024, 512, 256, 64,64]
        self.n_layers = len(self.dims)-1
        
        ## ------------Transition Layer------
        self.trans_in_list = self.dims[:-1][::-1]
        self.trans_out_list = [opt.ggd_ch] * 5
        
        self.trans = Transition(
            in_ch_list=self.trans_in_list,
            out_ch_list=self.trans_out_list,
        )
        ## ------------Attention Layer-----------
        self.attention_layers= nn.ModuleDict() 
        for i in range(self.n_layers):
            if self.opt.mda_mode == 'vanilla':
                print("Using vanilla mda!")
            elif 'mask' in self.opt.mda_mode:
                print("Using learnable mask mda!")
            self.attention_layers['mda_{}'.format(i)] = MaskguidedDualAttention(opt.ggd_ch, mask_mode=self.opt.mda_mode)
        #  ------------ Decoder Layer-----------  
        self.decoder_layers = nn.ModuleDict() 
        self.decoder_layers['deconv'] = GGD(opt.ggd_ch)
        
    def forward(self,z):
        x = z['skips']
        mda_masks = []
        ## -------------Layer Fusion-------
        x = self.trans(x)
        ## -------------Attention ------
        for i in range(self.n_layers-1, -1, -1):
            fused_layer = x[i]
            fused_layer, m = self.attention_layers['mda_{}'.format(i)](fused_layer)
            dst_shape = tuple(x[0].shape[2:])
            m = F.interpolate(m, size=dst_shape, mode='bilinear')
            mda_masks.append(m)
            x[i] = fused_layer
        ## ------------Decoding --------
        x = self.decoder_layers['deconv'](x).sigmoid()
        if self.opt.mda_mode != 'vanilla':
            return {"mask":[x]+mda_masks}
        else:
            return {"mask":[x]}
                




