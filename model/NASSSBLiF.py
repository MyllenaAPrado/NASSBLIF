import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from model.sswin import SwinTransformer
from einops import rearrange
from model.NAT import nat_mini


class NASSSBLiF(nn.Module):

    def __init__(self):
        super(NASSSBLiF, self).__init__()
        self.nat = nat_mini(pretrained=True, num_classes=1)
        self.angularFeatures = AngularFeatures()
        self.SFE = nn.Conv2d(
            3, 3, kernel_size=3, stride=1, dilation=5, padding=5, bias=False
        )
        self.AFE = nn.Conv2d(3, 3, kernel_size=5, stride=5, padding=0, bias=False)

        self.rerange_layer = Rearrange("b c h w -> b (h w) c")
        self.avg_pool = nn.AdaptiveAvgPool2d(5)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
        )

        self.conv_attent = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        x_ang = self.AFE(x)
        x_ang = self.angularFeatures(x_ang)
        x_ang = self.avg_pool(x_ang)

        x_spa = self.SFE(x)
        _, block2, _, block4 = self.nat(x_spa)
        b2 = self.avg_pool(rearrange(block2, "b h w c -> b c h w"))
        b4 = self.avg_pool(rearrange(block4, "b h w c -> b c h w"))

        feat = torch.cat((b2, b4, x_ang), dim=1)

        q = self.conv(feat)
        k = self.conv_attent(feat)
        pred = (q * k).sum(dim=2).sum(dim=2) / k.sum(dim=2).sum(dim=2)

        return pred


class AngularFeatures(nn.Module):

    def __init__(self):
        super(AngularFeatures, self).__init__()
        embed_dim = 256
        self.conv = nn.Conv2d(3, embed_dim, 3, 3)

        self.swintransformer1 = SwinTransformer(
            patches_resolution=(14, 14),
            depths=[2, 2, 2],
            num_heads=[4, 4, 4],
            embed_dim=embed_dim,
            window_size=2,
            dim_mlp=64,
            scale=0.2,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.swintransformer1(x)
        return x


if __name__ == "__main__":
    net = Network().cuda()
    from thop import profile

    input1 = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(net, inputs=(input1,))
    print("   Number of parameters: %.5fM" % (params / 1e6))
    print("   Number of FLOPs: %.5fG" % (flops / 1e9))
