import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import ConvBNReLU
import math

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


class PrimaryCaps_horizontal(nn.Module):
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps_horizontal, self).__init__()
        #  
        self.B = B
        self.P = P
        self.pose = nn.Conv2d(in_channels=A,  # 
                              out_channels=B * P * P,  # B: number of types of capsules  P: size of pose matrix is P*P
                              # B*P*P : 
                              kernel_size=K,  # capsule : 
                              stride=stride,
                              bias=True)  # [14,14,512] 32*4*4=512
        #  
        self.a = nn.Conv2d(in_channels=A,  # 
                           out_channels=B,  # 
                           kernel_size=K,
                           stride=stride,
                           bias=True)  # [14,14,32]
        self.conv12 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1, stride=1).cuda()
        self.conv24 = nn.Conv2d(in_channels=24, out_channels=1, kernel_size=1, stride=1).cuda()
        self.conv48 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, stride=1).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 
        p = self.pose(x)

        a = self.a(x)
        b, c, w, h = x.shape  #5, 32, h, w
        
        p = p.permute(0, 3, 2, 1).contiguous()
        a = a.permute(0, 3, 2, 1).contiguous()
      ##################
        if w == 12:
            p = self.conv12(p)
            a = self.conv12(a)
        if w == 24:
            p = self.conv24(p)
            a = self.conv24(a)
        if w == 48:
            p = self.conv48(p)
            a = self.conv48(a)         
            ###########################
        p = p.permute(0, 3, 2, 1).contiguous()
        
        
        a = a.permute(0, 3, 2, 1).contiguous()
        a = self.sigmoid(a)

        out = torch.cat([p, a], dim=1)  # 
        out = out.permute(0, 2, 3, 1).contiguous()  # 
       
        return out  # pose tensor and Activation tensor


class PrimaryCaps_vertical(nn.Module):
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps_vertical, self).__init__()
        #  
        self.B = B
        self.P = P
        self.pose = nn.Conv2d(in_channels=A,  # 
                              out_channels=B * P * P,  # B: number of types of capsules  P: size of pose matrix is P*P
                              # B*P*P : 
                              kernel_size=K,  # capsule : 
                              stride=stride,
                              bias=True)  # [14,14,512] 32*4*4=512
        #  
        self.a = nn.Conv2d(in_channels=A,  # 
                           out_channels=B,  # 
                           kernel_size=K,
                           stride=stride,
                           bias=True)  # [14,14,32]
        self.conv12 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1, stride=1).cuda()
        self.conv24 = nn.Conv2d(in_channels=24, out_channels=1, kernel_size=1, stride=1).cuda()
        self.conv48 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, stride=1).cuda()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)
        

    def forward(self, x):  # 
        p = self.pose(x)

        a = self.a(x)
        
        b, c, w, h = x.shape

        p = p.permute(0, 2, 3, 1).contiguous()
        a = a.permute(0, 2, 3, 1).contiguous()
        ###################
        if h == 12:
            p = self.conv12(p)
            a = self.conv12(a)
        if h == 24:
            p = self.conv24(p)
            a = self.conv24(a)
        if h == 48:
            p = self.conv48(p)
            a = self.conv48(a)
            ############################
        p = p.permute(0, 3, 1, 2).contiguous()

        #a = F.relu(self.bn(conv(a)))
        a = a.permute(0, 3, 1, 2).contiguous()
        a = self.sigmoid(a)
      
        out = torch.cat([p, a], dim=1)  # 
        out = out.permute(0, 2, 3, 1).contiguous()  # 

        return out  # pose tensor and Activation tensor


class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """

    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3,
                 coor_add=False, w_shared=False, horizontal=False, vertical=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.horizontal = horizontal
        self.vertical = vertical
        self.B = B  # input number of capsules
        self.C = C  # output number of capsules
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.iters = iters  # 
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        # Out[100]: tensor([1.8379], device='cuda:0')
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2 * math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))  # 
        self.beta_a = nn.Parameter(torch.zeros(C))  # 
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K * K * B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):  # 
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in  # 
        r = r / (r.sum(dim=2, keepdim=True) + eps)  # r[2] = C
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)  # 
        coeff = coeff.view(b, B, C, 1)  # 

        mu = torch.sum(coeff * v, dim=1, keepdim=True)  # 
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=1, keepdim=True) + eps  # 

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum  # 

        a_out = self.sigmoid(self._lambda * (self.beta_a - cost_h.sum(dim=2)))  # a_out --> aj
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):  # 
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu) ** 2 / (2 * sigma_sq) \
                   - torch.log(sigma_sq.sqrt()) \
                   - 0.5 * self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1. / C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out, r

    def add_pathes_hor(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """

        b, w, h, c = x.shape
        # assert h == w
        assert c == B * (psize + 1)
        oh = 1
        ow = math.ceil((w - K + 1) / stride)
        idx_s = [[(w_idx + k_idx) \
                  for w_idx in range(0, w - K + 1, stride)] \
                 for k_idx in range(0, K)]
        idxs_1 = [[(h_idx) \
                   for h_idx in range(0, 1, 1)] \
                  for k_idx in range(0, K)]
        x = x[:, idx_s, :, :]
        x = x[:, :, :, idxs_1, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x, oh, ow

    def add_pathes_ver(self, x1, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """

        b, w, h, c = x1.shape
        # assert h == w
        assert c == B * (psize + 1)
        ow = 1
        oh = math.ceil((h - K + 1) / stride)
        id_xs = [[(h_idx + k_idx) \
                  for h_idx in range(0, h - K + 1, stride)] \
                 for k_idx in range(0, K)]
        idxs_2 = [[(w_idx) \
                   for w_idx in range(0, 1, 1)] \
                  for k_idx in range(0, K)]
        x1 = x1[:, :, id_xs, :]

        x1 = x1[:, idxs_2, :, :, :]
        x1 = x1.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x1, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P * P

        # self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        # w = self.w

        x = x.view(b, B, 1, P, P)
        if w_shared:  # w_shared:share transformation matrix across w*h
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)  # 
        v = v.view(b, B, C, P * P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        # assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1. * torch.arange(h) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h * w * B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape  # x.shape = (*, h,  w, B*(P*P+1))
        if not self.w_shared:
            # add patches
            if self.horizontal:
                x, oh, ow = self.add_pathes_hor(x, self.B, self.K, self.psize, 1)
            if self.vertical:
                x, oh, ow = self.add_pathes_ver(x, self.B, self.K, self.psize, 1)

            # transform view
            p_in = x[:, :, :, :, :, :self.B * self.psize].contiguous()  # 
            a_in = x[:, :, :, :, :, self.B * self.psize:].contiguous()  # 
            p_in = p_in.view(b * oh * ow, self.K * self.K * self.B, self.psize)
            a_in = a_in.view(b * oh * ow, self.K * self.K * self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out, r = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C * self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
            
        else:
            assert c == self.B * (self.psize + 1)
            assert 1 == self.K
            assert 1 == self.stride
            p_in = x[:, :, :, :self.B * self.psize].contiguous()
            p_in = p_in.view(b, h * w * self.B, self.psize)
            a_in = x[:, :, :, self.B * self.psize:].contiguous()
            a_in = a_in.view(b, h * w * self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:  # coor_add: use scaled coordinate addition or not
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)
        out = out.permute(0,2,1,3).contiguous()
        
        return out,r


class CapsNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """

    def __init__(self, A=32, B=8, C=8, D=32, E=10, K=3, P=4, iters=3, inch=256):
        super(CapsNet, self).__init__()
        # image --> ReLU_conv1
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=A,
                               kernel_size=3, stride=1, padding=1).cuda()  # [32,32,1] --> [14,14,32]
        self.bn1 = nn.BatchNorm2d(num_features=A,  #
                                  eps=0.001,  #
                                  momentum=0.1,  #
                                  affine=True  #
                                  ).cuda()
        self.relu1 = nn.ReLU(inplace=False).cuda()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=A,
                               kernel_size=3, stride=1, padding=1).cuda()  # [32,32,1] --> [14,14,32]
        self.bn2 = nn.BatchNorm2d(num_features=A,  #
                                  eps=0.001,  #
                                  momentum=0.1,  #
                                  affine=True  #
                                  ).cuda()
        self.relu2 = nn.ReLU(inplace=False).cuda()

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=A,
                               kernel_size=3, stride=1, padding=1).cuda()  # [32,32,1] --> [14,14,32]
        self.bn3 = nn.BatchNorm2d(num_features=A,  #
                                  eps=0.001,  #
                                  momentum=0.1,  #
                                  affine=True  #
                                  ).cuda()
        self.relu3 = nn.ReLU(inplace=False).cuda()

        self.primary_caps_horizontal_r = PrimaryCaps_horizontal(A, B, 1, P, stride=1).cuda()
        self.primary_caps_horizontal_d = PrimaryCaps_horizontal(A, B, 1, P, stride=1).cuda()
        self.primary_caps_horizontal_t = PrimaryCaps_horizontal(A, B, 1, P, stride=1).cuda()

        self.conv_caps1_horizontal = ConvCaps(3 * B, C, 1, P, stride=2, iters=iters, horizontal=True,
                                              vertical=False).cuda()

        self.primary_caps_vertical_r = PrimaryCaps_vertical(A, B, 1, P, stride=1).cuda()
        self.primary_caps_vertical_d = PrimaryCaps_vertical(A, B, 1, P, stride=1).cuda()
        self.primary_caps_vertical_t = PrimaryCaps_vertical(A, B, 1, P, stride=1).cuda()
        self.conv_caps1_vertical = ConvCaps(3 * B, C, 1, P, stride=2, iters=iters, horizontal=False,
                                            vertical=True).cuda()

        self.convbnrelu = ConvBNReLU(nIn=3*A+C*17, nOut=256)
        self.drop = nn.Dropout(p=0.5)
        self.B = B

        self.convbnrelu_rs = ConvBNReLU(nIn=A+C*17, nOut=256)
        self.convbnrelu_ds = ConvBNReLU(nIn=A+C*17, nOut=256)
        self.convbnrelu_ts = ConvBNReLU(nIn=A+C*17, nOut=256)
        
        self.convbnrelu_rdts = ConvBNReLU(nIn=3*inch, nOut=256)
        
        
        self.convbnrelu_cath = ConvBNReLU(nIn=C*17*3, nOut=C*17*3)
        self.convbnrelu_catv = ConvBNReLU(nIn=C*17*3, nOut=C*17*3)
        



        self.convbnrelu_fr = ConvBNReLU(nIn=512, nOut=256)
        self.convbnrelu_fd = ConvBNReLU(nIn=512, nOut=256)
        self.convbnrelu_ft = ConvBNReLU(nIn=512, nOut=256)

        self.convbnrelu_wr =  nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1).cuda(0)
        self.convbnrelu_wd =  nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1).cuda(0)
        self.convbnrelu_wt =  nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1).cuda(0)


    def forward(self, x, y, z):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)

        z = self.conv3(z)
        z = self.bn3(z)
        z = self.relu3(z)

        h_d_r = self.primary_caps_horizontal_r(x)  # bsz,h, w, 136
        h_d_d = self.primary_caps_horizontal_d(y)
        h_d_t = self.primary_caps_horizontal_t(z)
        #h_d, h_r = self.conv_caps1_horizontal(torch.cat([h_d_r, h_d_d, h_d_t], dim=3))  # bsz, 1, w, 136
        h_d_cat = torch.cat([h_d_r, h_d_d, h_d_t], dim=3).permute(0,3,1,2)
        h_d, h_r = self.conv_caps1_horizontal(self.convbnrelu_cath(h_d_cat).permute(0,2,3,1))        #bsz, h, 1, 136



        v_d_r = self.primary_caps_vertical_r(x)
        v_d_d = self.primary_caps_vertical_d(y)
        v_d_t = self.primary_caps_vertical_t(z)
        #v_d, v_r = self.conv_caps1_vertical(torch.cat([v_d_r, v_d_d, v_d_t], dim=3))
        v_d_cat = torch.cat([v_d_r, v_d_d, v_d_t], dim=3).permute(0,3,1,2)
        v_d , v_r= self.conv_caps1_vertical(self.convbnrelu_catv(v_d_cat).permute(0,2,3,1))     

        h_d = h_d.permute(0, 3, 1, 2).contiguous()
        v_d = v_d.permute(0, 3, 1, 2).contiguous()
        out = torch.matmul(h_d, v_d)
        out = self.convbnrelu(torch.cat([x, y, z, out], dim=1))  # bsz, 256, h, w
        # out = self.drop(out)

        bsz, d_h, h_h, w_h  = h_d.shape
        bsz, d_v,  h_v, w_v  = v_d.shape

        _, c1_h, c2_h = h_r.shape
        _, c1_v, c2_v = v_r.shape


        h_r = torch.reshape(h_r, [bsz, h_h, w_h, c1_h, c2_h])
        v_r = torch.reshape(v_r, [bsz, h_v, w_v, c1_v, c2_v])
        h_r_split = torch.split(h_r, self.B, dim=3)
        v_r_split = torch.split(v_r, self.B, dim=3)

        img_hr = h_r_split[0].mean(dim=4)  #5, 12, 1, 8
        depth_hr = h_r_split[1].mean(dim=4)
        thermal_hr = h_r_split[2].mean(dim=4)

        img_vr = v_r_split[0].mean(dim=4)  #5, 1, 12, 8
        depth_vr = v_r_split[1].mean(dim=4)
        thermal_vr = v_r_split[2].mean(dim=4)

        img_ratio_h = (torch.exp(img_hr) / (
                    torch.exp(img_hr) + torch.exp(depth_hr) + torch.exp(thermal_hr)) + 1e-8).repeat(1, 1, 1,17)  # 5,  12, 1, 136
        depth_ratio_h = (torch.exp(depth_hr) / (
                    torch.exp(img_hr) + torch.exp(depth_hr) + torch.exp(thermal_hr)) + 1e-8).repeat(1, 1, 1, 17)
        thermal_ratio_h = (torch.exp(thermal_hr) / (
                    torch.exp(img_hr) + torch.exp(depth_hr) + torch.exp(thermal_hr)) + 1e-8).repeat(1, 1, 1, 17)
        img_ratio_v = (torch.exp(img_vr) / (
                    torch.exp(img_vr) + torch.exp(depth_vr) + torch.exp(thermal_vr)) + 1e-8).repeat(1, 1, 1,17)  # 5,  1, 12, 136
        depth_ratio_v = (torch.exp(depth_vr) / (
                    torch.exp(img_vr) + torch.exp(depth_vr) + torch.exp(thermal_vr)) + 1e-8).repeat(1, 1, 1, 17)
        thermal_ratio_v = (torch.exp(thermal_vr) / (
                    torch.exp(img_vr) + torch.exp(depth_vr) + torch.exp(thermal_vr)) + 1e-8).repeat(1, 1, 1, 17)

        h_d_rs = torch.mul(h_d_r, img_ratio_h).permute(0, 3, 1, 2).contiguous()  # 5, 136 , 12, 1
        h_d_ds = torch.mul(h_d_d, depth_ratio_h).permute(0, 3, 1, 2).contiguous()
        h_d_ts = torch.mul(h_d_t, thermal_ratio_h).permute(0, 3, 1, 2).contiguous()

        v_d_rs = torch.mul(v_d_r, img_ratio_v).permute(0, 3, 1, 2).contiguous()  # 5, 136, 1, 12
        v_d_ds = torch.mul(v_d_d, depth_ratio_v).permute(0, 3, 1, 2).contiguous()
        v_d_ts = torch.mul(v_d_t, thermal_ratio_v).permute(0, 3, 1, 2).contiguous()

        rs = torch.matmul(h_d_rs, v_d_rs)
        ds = torch.matmul(h_d_ds, v_d_ds)
        ts = torch.matmul(h_d_ts, v_d_ts)
        # print(ts.shape)

        out_rs =  self.convbnrelu_rs(torch.cat([x,  rs], dim=1))    #bsz, 256, h, w
        out_ds =  self.convbnrelu_ds(torch.cat([y,  ds], dim=1)) 
        out_ts =  self.convbnrelu_ts(torch.cat([z,  ts], dim=1)) 

        f_r = self.convbnrelu_fr(torch.cat([out,  out_rs], dim=1))
        f_d = self.convbnrelu_fd(torch.cat([out,  out_ds], dim=1))
        f_t = self.convbnrelu_ft(torch.cat([out,  out_ts], dim=1))


        weit_rs = torch.sigmoid(self.convbnrelu_wr(out_rs))  #bsz, 1, h, w
        weit_ds = torch.sigmoid(self.convbnrelu_wd(out_ds))
        weit_ts = torch.sigmoid(self.convbnrelu_wt(out_ts))

        f_r_new = torch.mul(f_r,weit_rs) + f_r
        f_d_new = torch.mul(f_d,weit_ds) + f_d
        f_t_new = torch.mul(f_t,weit_ts) + f_t


        #out_rdts = self.convbnrelu_rdts(torch.cat([out_rs, out_ds, out_ts], dim=1))

        out_rdts = self.convbnrelu_rdts(torch.cat([f_r_new, f_d_new, f_t_new], dim=1))


        return out, out_rdts


def PWRF(**kwargs):
    """
    Constructs a CapsNet model.
    """
    model = CapsNet(**kwargs)
    return model


'''
TEST
Run this code with:
'''
if __name__ == '__main__':
    x = torch.randn([2, 128, 88, 88]).cuda()
    model = DCR(E=10).cuda()
    output1 = model(x)
    print(output1.shape)
