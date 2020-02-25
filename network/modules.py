import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Identity(nn.Module):
    # a dummy identity module
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, stride=2):
        super(Unpool, self).__init__()

        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:, :, 0, 0] = 1

    def forward(self, x):
        assert x.dim() == 4
        num_channels = x.size(1)
        return F.conv_transpose2d(x,
                                  self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1),
                                  stride=self.stride, groups=num_channels)


def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules.children():
            weights_init(m)
            # if isinstance(m, nn.Conv2d):
            #     if type == 'xavier':
            #         torch.nn.init.xavier_normal_(m.weight)
            #     elif type == 'kaiming':  # msra
            #         torch.nn.init.kaiming_normal_(m.weight)
            #     else:
            #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #         m.weight.data.normal_(0, math.sqrt(2. / n))
            #
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # elif isinstance(m, nn.ConvTranspose2d):
            #     if type == 'xavier':
            #         torch.nn.init.xavier_normal_(m.weight)
            #     elif type == 'kaiming':  # msra
            #         torch.nn.init.kaiming_normal_(m.weight)
            #     else:
            #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #         m.weight.data.normal_(0, math.sqrt(2. / n))
            #
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1.0)
            #     m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     if type == 'xavier':
            #         torch.nn.init.xavier_normal_(m.weight)
            #     elif type == 'kaiming':  # msra
            #         torch.nn.init.kaiming_normal_(m.weight)
            #     else:
            #         m.weight.data.fill_(1.0)
            #
            #     if m.bias is not None:
            #         m.bias.data.zero_()


# def weights_init(m):
#     # Initialize kernel weights with Gaussian distributions
#     if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.ConvTranspose2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()

def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    assert 2 * padding == kernel_size - 1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def depthwise(in_channels, kernel_size):
    padding = (kernel_size - 1) // 2  # Maintain resolution
    assert 2 * padding == kernel_size - 1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def convt(in_channels, out_channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"  # k must be odd
    return nn.Sequential(  # Double the resolution
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                           stride, padding, output_padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def convt_dw(channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
    return nn.Sequential(
        nn.ConvTranspose2d(channels, channels, kernel_size,
                           stride, padding, output_padding, bias=False, groups=channels),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


def deconv(in_channels, out_channels, kernel_size=5, output_size=None):
    modules = [convt_dw(in_channels, kernel_size), pointwise(in_channels, out_channels)]
    if output_size:
        modules.append(nn.UpsamplingNearest2d(size=output_size))
    return nn.Sequential(*modules)


def upconv(in_channels, out_channels, kernel_size=5, output_size=None):
    # Unpool then conv maintaining resolution

    modules = [
        Unpool(2),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]
    if output_size:
        modules.append(nn.UpsamplingNearest2d(size=output_size))
    return nn.Sequential(*modules)


class upproj(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels, output_size=None):
        super(upproj, self).__init__()
        self.unpool = Unpool(2)
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if output_size:
            modules.append(nn.UpsamplingNearest2d(size=output_size))
        self.branch1 = nn.Sequential(*modules)

        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
                   nn.BatchNorm2d(out_channels)
                   ]
        if output_size:
            modules.append(nn.UpsamplingNearest2d(size=output_size))
        self.branch2 = nn.Sequential(*modules)

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return F.relu(x1 + x2)


class shuffle_conv(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=None):
        super(shuffle_conv, self).__init__()
        modules = [depthwise(in_channels, 5), pointwise(in_channels, out_channels)]
        if output_size:
            modules.append(nn.UpsamplingNearest2d(size=output_size))
        self.conv = nn.Sequential(
            *modules
        )

    def forward(self, x):
        x = F.pixel_shuffle(x, upscale_factor=2)
        return self.conv(x)


class Seq_Ex_Block(nn.Module):
    def __init__(self, in_ch, r):
        super(Seq_Ex_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        # print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
        return x.mul(se_weight)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:-2]), -1).mean(-1)


class ASPD(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_resolution=None, output_size=None):
        super(ASPD, self).__init__()
        dilation_rates = [1, 3, 5, 7]
        modules = [0] * len(dilation_rates)
        for i, dilation in enumerate(dilation_rates):
            kernel_size = 4
            padding = ((kernel_size - 1) * dilation - 1) // 2
            modules[i] = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, dilation=dilation, stride=2),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(out_channels, out_channels, 1),
                          nn.ReLU(inplace=True),
                          nn.Dropout2d(p=0.4)
                          ]

            if output_size:
                modules.append(nn.UpsamplingNearest2d(size=output_size))
        self.aspd1 = nn.Sequential(*modules[0])
        self.aspd2 = nn.Sequential(*modules[1])
        self.aspd3 = nn.Sequential(*modules[2])
        self.aspd4 = nn.Sequential(*modules[3])
        self.concat_process = nn.Sequential(
            # Seq_Ex_Block(out_channels * len(dilation_rates), len(dilation_rates)),
            ChannelwiseLocalAttention(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(out_channels * len(dilation_rates), out_channels, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.5)
        )
        weights_init(self.aspd1)
        weights_init(self.aspd2)
        weights_init(self.aspd3)
        weights_init(self.aspd4)
        # self.concat_process.apply(weights_init)
        weights_init(self.concat_process)
        # weights_init(self.modules(), type='xavier')

    def forward(self, x):
        x1 = self.aspd1(x)
        x2 = self.aspd2(x)
        x3 = self.aspd3(x)
        x4 = self.aspd4(x)
        output = torch.cat([x1, x2, x3, x4], dim=1)
        output = self.concat_process(output)
        return output


class ChannelwiseLocalAttention(nn.Module):
    def __init__(self, pooling_size=(4, 4), r=2):
        super(ChannelwiseLocalAttention, self).__init__()
        self.pooling_size = pooling_size
        # self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
        self.pool = nn.AdaptiveAvgPool2d(output_size=pooling_size)
        in_channels = pooling_size[0] * pooling_size[1]
        out_channels = in_channels // r
        # Each conv_matrix having shape of 1 x 1 x (H*W) x (H*W/r)
        # They will be convolved on channel-wise matrix of shape (H*W) * C
        self.conv_Q = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv_K = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x_avg = self.pool(x)
        N, C, H, W = x_avg.size()
        x_avg = x_avg.view(N, C, H * W)
        x_avg = x_avg.transpose(1, 2)  # Reshape to channel-wise vector at each x_avg[0,0,:]
        Q = self.conv_Q(x_avg)  # N x (H/r*W/r) x c
        K = self.conv_K(x_avg)  # N x (H/r*W/r) x c
        score = torch.matmul(Q.transpose(1, 2), K)
        score = F.softmax(score, dim=-1)
        att_weights = torch.matmul(score, x_avg.transpose(1, 2))  # att_weights = (C x C) x (C x (H*W)) = C x (H*W)

        # Repeat the attention weights by the stride of pooling layer
        # to transform weight_mask matching the shape of original input
        h_scale, w_scale = x.size(2) // self.pooling_size[0], x.size(3) // self.pooling_size[1]
        att_weights = att_weights.view(N, C, H * W, 1)
        att_weights = att_weights.repeat(1, 1, 1, w_scale)
        att_weights = att_weights.view(N, C, H, W * w_scale)
        att_weights = att_weights.repeat(1, 1, 1, h_scale)
        att_weights = att_weights.view(N, C, H * h_scale, W * w_scale)

        if att_weights.size() != x.size():
            att_weights = F.interpolate(att_weights, size=list(x.shape[2:]), mode='nearest')
        assert att_weights.size() == x.size()

        # Re-weight original input by weight mask
        return att_weights * x
