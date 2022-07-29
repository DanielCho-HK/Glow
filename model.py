import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, pi
import numpy as np
import scipy.linalg as la

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super(ActNorm, self).__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            std = torch.std(x, dim=(0, 2, 3), keepdim=True)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        _, _, height, width = x.shape

        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (x + self.loc), logdet
        else:
            return self.scale * (x + self.loc)

    def reverse(self, x):
        return x / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super(InvConv2d, self).__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        # 保证矩阵的可逆性
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        _, _, height, width = x.shape

        out = F.conv2d(x, self.weight)
        logdet = (
                height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
                  )
        return out, logdet

    def reverse(self, x):
        return F.conv2d(x, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super(InvConv2dLU, self).__init__()
        # 服从高斯分布
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)  # Q是正交矩阵
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))

        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))

        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        _, _, height, width = x.shape
        weight = self.calc_weight()
        out = F.conv2d(x, weight)

        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def reverse(self, x):
        weight = self.calc_weight()
        return F.conv2d(x, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv2d, self).__init__()
        # 参数只在初始化时候全为0
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, x):
        out = F.pad(x, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super(AffineCoupling, self).__init__()
        self.affine = affine
        # in_channel必须为偶数，否则报错
        self.net = nn.Sequential(
            nn.Conv2d(in_channel//2, filter_size, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel//2)
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=1)

        if self.affine:
            log_s, t = self.net(x_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (x_b + t) * s
            logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        else:
            net_out = self.net(x_a)
            out_b = x_b + net_out
            logdet = None

        return torch.cat([x_a, out_b], 1), logdet

    def reverse(self, x):
        x_a, x_b = x.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(x_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = x_b / s - t
        else:
            net_out = self.net(x_a)
            out_b = x_b - net_out
        return torch.cat([x_a, out_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super(Flow, self).__init__()
        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, x):
        out, logdet = self.actnorm(x)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, x):
        out = self.coupling.reverse(x)
        out = self.invconv.reverse(out)
        out = self.actnorm.reverse(out)

        return out


def gaussian_log_p(x, mean, log_std):
    return -0.5 * log(2 * pi) - log_std - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_std)


def gaussian_sample(eps, mean, log_std):
    return mean + torch.exp(log_std) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super(Block, self).__init__()
        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split
        if self.split:
            self.prior = ZeroConv2d(in_channel*2, in_channel*4)
        else:
            self.prior = ZeroConv2d(in_channel*4, in_channel*8)

    def forward(self, x):
        b_size, n_channel, height, width = x.shape
        squeezed = x.view(b_size, n_channel, height//2, 2, width//2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_std = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_std = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, x, eps=None, reconstruct=False):
        out = x

        if reconstruct:
            if self.split:
                out = torch.cat([x, eps], 1)

            else:
                out = eps

        else:
            if self.split:
                mean, log_std = self.prior(out).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                out = torch.cat([x, z], 1)

            else:
                zero = torch.zeros_like(out)
                mean, log_std = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                out = z

        for flow in self.flows[::-1]:
            out = flow.reverse(out)

        b_size, n_channel, height, width = out.shape

        unsqueezed = out.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super(Glow, self).__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel

        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, x):
        log_p_sum = 0
        logdet = 0
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                out = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                out = block.reverse(out, z_list[-(i + 1)], reconstruct=reconstruct)

        return out
