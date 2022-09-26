import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def double_conv3d(in_ch, out_ch):
    return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
            )


def double_conv2d(in_ch, out_ch):
    return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )


def double_downconv3d(in_ch, out_ch):
    return nn.Sequential(nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
                        double_conv3d(in_ch, out_ch))


class double_upconv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv2d(in_ch//2+out_ch, out_ch)

    def forward(self, x1, x2):
        # x1: [c, h/2], x2: [c/2, h], catted: [c/2+c/2, h]
        # return [c/2, h]
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Generator3(nn.Module):
    def __init__(self, in_ch, out_ch, cfg):
        super().__init__()
        n_kernels = cfg.n_kernels       # [32, 64, 128]
        bn_kernel = cfg.bn_kernel       # 128
        self.n_layers = len(n_kernels)  # 3

        self.img_size = cfg.resize_h
        self.inp = double_conv3d(in_ch, n_kernels[0])
        self.down1 = double_downconv3d(n_kernels[0], n_kernels[1])
        self.down2 = double_downconv3d(n_kernels[1], n_kernels[2])
        self.global_pool = nn.AdaptiveMaxPool3d((1, self.img_size//(2**(self.n_layers-1)), self.img_size//(2**(self.n_layers-1))))
        self.bottleneck = double_conv2d(n_kernels[2], bn_kernel)
        self.up1 = double_upconv2d(bn_kernel, n_kernels[1])
        self.up2 = double_upconv2d(n_kernels[1], n_kernels[0])
        self.outc = nn.Conv2d(n_kernels[0], out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inp(x)                # [32, h]
        x2 = self.down1(x1)             # [64, h/2]
        bottleneck = self.down2(x2)     # [128, h/4]
        bottleneck = self.global_pool(bottleneck).squeeze(2)
        bottleneck = self.bottleneck(bottleneck) # [128, h/4]
        # x1: [128, h/4], x2: [64, h/2], x: [64, h/2]
        x = self.up1(bottleneck, F.adaptive_max_pool3d(x2,
                        (1, self.img_size//(2**(self.n_layers-2)), self.img_size//(2**(self.n_layers-2)))).squeeze(2))
        # x1: [64, h/2], x2: [32, h], x: [64, h]
        x = self.up2(x, F.adaptive_max_pool3d(x1,
                        (1, self.img_size//(2**(self.n_layers-3)), self.img_size//(2**(self.n_layers-3)))).squeeze(2))
        x = self.outc(x)
        return x


class DAttention(nn.Module):
    '''
    key: [B, nframe, X]
    query: [B, 1, X]
    value: [B, nframe, C, H, W]
    dot product attention
    '''
    def __init__(self, k_dim, q_dim, att_dim):
        super().__init__()
        # reduction_ratio = cfg.reduction_ratio
        self.mlp_k = nn.Sequential(
                nn.Flatten(start_dim=2),
                nn.Linear(k_dim, att_dim),
                nn.ReLU(inplace=True),
                nn.Linear(att_dim, att_dim)
            )
        self.mlp_q = nn.Sequential(
                nn.Flatten(start_dim=2),
                nn.Linear(q_dim, att_dim),
                nn.ReLU(inplace=True),
                nn.Linear(att_dim, att_dim)
            )
        self._init_weights()

    def _init_weights(self):
        layers = [self.mlp_k.parameters(), self.mlp_q.parameters()]
        for layer in layers:
            for p in layer:
                if p.dim() > 1:
                    # nn.init.xavier_uniform_(p)
                    nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')

    def forward(self, key, query):
        '''
        Inputs:
            key: [B, C, D, H, W]
            query: [B, C, H, W]
        Outputs:
            value: [B, C, H, W] (attention-pooled, i.e., weighted sum)
        '''
        key = key.transpose(1, 2) # [B, D, C, H, W]
        query = query.unsqueeze(1) # [B, 1, C, H, W]
        
        key_proj = self.mlp_k(key) # [B, D, att_dim]
        query_proj = self.mlp_q(query) # [B, 1, att_dim]

        # scores = torch.matmul(key_proj, query_proj.transpose(1, 2)) # [B, D, 1]
        scores = torch.matmul(key_proj, query_proj.transpose(-2, -1)) # [B, D, 1]
        attn = F.softmax(scores, dim=1) # [B, D, 1]

        value = key*attn[..., None, None] # [B, D, C, H, W]
        value = value.sum(dim=1).squeeze(1) # [B, C, H, W]
        return value


class Generator4(nn.Module):
    def __init__(self, in_ch, out_ch, cfg):
        super().__init__()
        n_kernels = cfg.n_kernels
        bn_kernel = cfg.bn_kernel
        att_dim = cfg.att_dim
        self.n_layers = len(n_kernels) # 4, [32, 64, 64, 128]
        bn_hw_size = cfg.resize_h//(2**(self.n_layers-1))

        self.inp = double_conv3d(in_ch, n_kernels[0])
        self.down1 = double_downconv3d(n_kernels[0], n_kernels[1])
        self.down2 = double_downconv3d(n_kernels[1], n_kernels[2])
        self.down3 = double_downconv3d(n_kernels[2], n_kernels[3])
        self.global_pool = nn.AdaptiveMaxPool3d((1, bn_hw_size, bn_hw_size))
        self.bottleneck = double_conv2d(n_kernels[3], bn_kernel)
        self.pool1 = DAttention(k_dim=n_kernels[2]*bn_hw_size*2*bn_hw_size*2,
                                q_dim=n_kernels[3]*bn_hw_size*bn_hw_size,
                                att_dim=att_dim)
        self.up1 = double_upconv2d(bn_kernel, n_kernels[2])
        self.pool2 = DAttention(k_dim=n_kernels[1]*bn_hw_size*4*bn_hw_size*4,
                                q_dim=n_kernels[2]*bn_hw_size*2*bn_hw_size*2,
                                att_dim=att_dim)
        self.up2 = double_upconv2d(n_kernels[2], n_kernels[1])
        self.pool3 = DAttention(k_dim=n_kernels[0]*bn_hw_size*8*bn_hw_size*8,
                                q_dim=n_kernels[1]*bn_hw_size*4*bn_hw_size*4,
                                att_dim=att_dim)
        self.up3 = double_upconv2d(n_kernels[1], n_kernels[0])
        self.outc = nn.Conv2d(n_kernels[0], out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inp(x)       # [32, h], [1,32,4,256,256]
        x2 = self.down1(x1)    # [64, h/2], [1,64,4,128,128]
        x3 = self.down2(x2)     # [64, h/4], [1,64,4,64,64]
        bottleneck = self.down3(x3)     # [128, h/8], [1,128,4,32,32]
        bottleneck = self.global_pool(bottleneck).squeeze(2) # [1,128,32,32]
        bottleneck = self.bottleneck(bottleneck) # [128, h/8], [1,128,32,32]
        # x1: [128, h/8], x2: [64, h/4], [1,64,64,64]
        pooled3 = self.pool1(x3, bottleneck)
        x = self.up1(bottleneck, pooled3)
        # x1: [64, h/4], x2: [64, h/2], [1,64,128,128]
        pooled2 = self.pool2(x2, x)
        x = self.up2(x, pooled2)
        # x1: [64, h/2], x2: [64, h/1], [1,32,256,256]
        pooled1 = self.pool3(x1, x)
        x = self.up3(x, pooled1)
        x = self.outc(x)
        return x


class Config: # for debug
    n_kernels = [32, 64, 64, 128]
    bn_kernel = n_kernels[-1]
    resize_h = 256
    att_dim = 50


if __name__=="__main__":
    cfg = Config()
    in_channels = 3
    out_channels = 3
    rand_input = torch.rand([1, 3, 4, cfg.resize_h, cfg.resize_h])
    model = Generator4(in_channels, out_channels, cfg)

    print('model')
    print(model)
    print('input shape', rand_input.size())
    out = model(rand_input)
    # out = model(rand_input)
    print('out shape', out.size())