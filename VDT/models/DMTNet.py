﻿""""
backbone is SwinTransformer
"""
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from models.SwinT import SwinTransformer

from models.PWRF import PWRF
#from models.EMRouting import PrimaryCaps,ConvCaps
import torch.nn.functional as F
import os
import onnx
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class DMTNet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(DMTNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.thermal_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.afem = AFEM(1024, 1024)  # 注意力特征提取模块


        self.decoder1 = Decoder()
        self.decoder2 = Decoder()




        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        # self.conv2048_1024 = conv3x3_bn_relu(2048, 1024)
        # self.conv1024_512 = conv3x3_bn_relu(1024, 512)
        # self.conv512_256 = conv3x3_bn_relu(512, 256)
        self.conv256_32 = conv3x3_bn_relu(256, 32)
        #self.conv64_1 = conv3x3(64, 1)
        self.conv32_1 = conv3x3(32, 1)
        self.edge_layer = Edge_Module()
        self.edge_feature = conv3x3_bn_relu(1, 32)
        self.fuse_edge_sal = conv3x3(32, 1)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            conv3x3(32, 1)
        )

        self.relu = nn.ReLU(True)

        ##########capsule#####
        #self.primary_caps_r1 = PrimaryCaps(1024, 16, 1, 4, stride=1)
        #self.primary_caps_d1 = PrimaryCaps(1024, 16, 1, 4, stride=1)
        #self.conv_caps_rd1 = ConvCaps(32, 32, 1, 4, stride=1,iters=2)
        #self.conv_544_2048 = nn.Conv2d(544, 2048, kernel_size=1, stride=1, padding=0)

        self.PWRF1 = PWRF()
        self.PWRF2 = PWRF()
        self.PWRF3 = PWRF()


        '''
        self.conv1024_256r =  conv3x3(1024, 256)
        self.conv512_256r   =  conv3x3(512, 256)
        self.conv1024_256d =  conv3x3(1024, 256)
        self.conv512_256d   =  conv3x3(512, 256)
        self.conv1024_256t =  conv3x3(1024, 256)
        self.conv512_256t   =  conv3x3(512, 256)


        '''
        self.conv1024_256r =  conv3x3_bn_relu(1024, 256)
        self.conv512_256r   =  conv3x3_bn_relu(512, 256)
        self.conv1024_256d =  conv3x3_bn_relu(1024, 256)
        self.conv512_256d   =  conv3x3_bn_relu(512, 256)
        self.conv1024_256t =  conv3x3_bn_relu(1024, 256)
        self.conv512_256t   =  conv3x3_bn_relu(512, 256)


        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=2048, kernel_size=3, stride=1, padding=1)
        #self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv256_1024 =  conv3x3_bn_relu( 256,1024)
        self.conv256_512   =  conv3x3_bn_relu( 256,512)
        self.conv384_256   =  conv3x3_bn_relu( 384,256)

        self.conv256_2048   =  conv3x3_bn_relu( 256, 2048)
        self.conv256_1024   =  conv3x3_bn_relu( 256, 1024)
        self.conv256_512     =  conv3x3_bn_relu( 256, 512)




    def forward(self, x, d, t):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)
        thermal_list =self.thermal_swin(t)

        r4 = rgb_list[0]  # 128*96
        r3 = rgb_list[1]  # 256*48
        r2 = rgb_list[2]  # 512*24
        r1 = rgb_list[3]  # 1024*12
        #r1 = self.afem(r1)      # [bsz, 1024, 12, 12]

        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]
        #d1 = self.afem(d1)   #[bsz, 128, 12, 12]

        t4 = thermal_list[0]
        t3 = thermal_list[1]
        t2 = thermal_list[2]
        t1 = thermal_list[3]




        fuse_shd1 ,fuse_spc1 = self.PWRF1(self.conv1024_256r(r1),  self.conv1024_256d(d1),  self.conv1024_256t(t1)) #bsz, 256, 12, 12
        fuse_shd2 ,fuse_spc2 = self.PWRF2(self.conv512_256r(  r2)  , self.conv512_256d( d2),   self.conv512_256t( t2)) #bsz, 256, 24, 24
        fuse_shd3 ,fuse_spc3 = self.PWRF3( r3, d3 ,t3)   #bsz, 256, 48, 48


        # 融合特征
        #fuse1 = self.cmfm1(r1, d1)  # [2048]   , 12
        #fuse1_cap = self.cmfm1_cap(r1_cap_new,d1_cap_new)  # bsz, 544, 12, 12
        #fuse1_cap = self.conv_544_2048(fuse1_cap)
        #print(fuse1_cap.max())
        #fuse1 +=  fuse1_cap
        #fuse2 = self.cmfm2(r2, d2)  # [1024]   , 24
        #fuse3 = self.cmfm3(r3, d3)  # [512]     , 48
        #fuse4 = self.cmfm4(r4, d4)  # [256]     , 96
        #end_fuse1, out43, out432 = self.decoder(fuse1, fuse2, fuse3, fuse4)
        #end_fuse = self.decoder2(fuse1, out43, out432, end_fuse1, end_fuse1)

        fuse1 = self.deconv1(fuse_shd1)                            # [2048]   , 12
        fuse2 = self.conv256_1024(fuse_shd2)                    # [1024]   , 24
        fuse3 = self.conv256_512(fuse_shd3)                      # [512]     , 48
        fuse4 = self.conv384_256(torch.cat((r4, d4, t4),dim = 1) )        # [256]     , 96
        end_fuse1, out43, out432 = self.decoder1(fuse1, fuse2, fuse3, fuse4)
        end_fuse = self.decoder2(fuse1, out43, out432, end_fuse1, end_fuse1)

        fuse1_s = self.conv256_2048(fuse_spc1 )
        fuse2_s = self.conv256_1024(fuse_spc2 )
        fuse3_s = self.conv256_512(  fuse_spc3 )
        end_fuse1_s, out43_s, out432_s = self.decoder1(fuse1_s, fuse2_s, fuse3_s, fuse4)
        end_fuse_s = self.decoder2(fuse1_s, out43_s, out432_s ,end_fuse1_s, end_fuse1_s)



        edge_map = self.edge_layer(d4, d3)
        edge_feature = self.edge_feature(edge_map)
        end_sal = self.conv256_32(end_fuse)  # [b,32]
        end_sal1 = self.conv256_32(end_fuse1)
        up_edge = self.up_edge(edge_feature)
        #out1 = self.relu(torch.cat((end_sal1, edge_feature), dim=1))
        #out = self.relu(torch.cat((end_sal, edge_feature), dim=1))
        out = self.up4(end_sal)
        out1 = self.up4(end_sal1 )
        sal_out = self.conv32_1(out)
        sal_out1 = self.conv32_1(out1)


        end_sal_s = self.conv256_32(end_fuse_s)  # [b,32]
        end_sal1_s = self.conv256_32(end_fuse1_s)
        #out1_s = self.relu(torch.cat((end_sal1_s, edge_feature), dim=1))
        #out_s = self.relu(torch.cat((end_sal_s, edge_feature), dim=1))
        out_s = self.up4(end_sal_s)
        out1_s = self.up4(end_sal1_s)
        sal_out_s = self.conv32_1(out_s)
        sal_out1_s = self.conv32_1(out1_s)
        fin = sal_out  + sal_out1 + sal_out_s +sal_out1_s 


        return sal_out, up_edge, sal_out1, sal_out_s, sal_out1_s, fin


    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")
        self.thermal_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Thermal SwinTransformer loading pre_model ${pre_model}")

    # def load_pre(self, model):
    #     self.edge_layer.load_state_dict(model,strict=False)











class AFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(AFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim),
                                       nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)

        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:],
                           mode='bilinear')
        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))






class MAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MAM, self).__init__()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = conv3x3_bn_relu(in_ch, out_ch)
        self.aff = AFF(out_ch)

    def forward(self, fuse_high, fuse_low):
        fuse_high = self.up2(fuse_high)
        fuse_high = self.conv(fuse_high)
        fe_decode = self.aff(fuse_high, fuse_low)
        # fe_decode = fuse_high + fuse_low
        return fe_decode



# Cascaded Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.mam12 = MAM(2048, 1024)
        self.mam23 = MAM(1024, 512)
        self.mam34 = MAM(512, 256)
        self.conv256_512 = conv3x3_bn_relu(256, 512)
        self.conv256_1024 = conv3x3_bn_relu(256, 1024)
        self.conv256_2048 = conv3x3_bn_relu(256, 2048)

        """
        此处参数：fuse1,fuse2,fuse3,fuse4 特征等级依次升高，通道数逐渐升高，尺寸逐渐减小
        fuse4:2048,12,12
        fuse3:1024,24,24
        fuse2:512,48,48
        fuse1:256,96,96
        out为上一个解码器预测的:1,256,96,96
        """

    def forward(self, fuse4, fuse3, fuse2, fuse1, iter=None):
        if iter is not None:

            out_fuse4 = F.interpolate(iter, size=(12, 12), mode='bilinear')
            out_fuse4 = self.conv256_2048(out_fuse4)
            fuse4 = out_fuse4 + fuse4

            out_fuse3 = F.interpolate(iter, size=(24, 24), mode='bilinear')
            out_fuse3 = self.conv256_1024(out_fuse3)
            fuse3 = out_fuse3 + fuse3

            out_fuse2 = F.interpolate(iter, size=(48, 48), mode='bilinear')
            out_fuse2 = self.conv256_512(out_fuse2)
            fuse2 = out_fuse2 + fuse2

            fuse1 = iter + fuse1

            out43 = self.mam12(fuse4, fuse3)
            out432 = self.mam23(out43, fuse2)
            out4321 = self.mam34(out432, fuse1)
            return out4321
        else:
            out43 = self.mam12(fuse4, fuse3)  # [b,1024,24,24]
            out432 = self.mam23(out43, fuse2)  # [b,512,48,48]
            out4321 = self.mam34(out432, fuse1)  # [b,256,96,96]
            return out4321, out43, out432


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[128, 256], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifer = nn.Conv2d(mid_fea * 2, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 2)

    def forward(self, x2, x4):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        # xa = x * residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    net = CatNet()
    a = torch.randn([2, 3, 384, 384])
    b = torch.randn([2, 3, 384, 384])
    s, e, s1 = net(a, b)
    print("s.shape:", e.shape)
