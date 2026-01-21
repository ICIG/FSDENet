import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import math
from einops.layers.torch import Rearrange
from pytorch_wavelets import DWTForward
import numbers
from einops import rearrange

config = {
    "tiny":[64, 128, 256, 512],
    "small": [96, 192, 384, 768],
    "base": [128, 256, 512, 1024]
}

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    
class MakeLayerNorm(nn.Module):
    def __init__(self, dim):
        super(MakeLayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)





class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.1, 
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}



def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[64, 128, 256, 512],mode="tiny", **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"],strict=False)
    return model



def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],**kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model



def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], mode = "base",**kwargs,)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=16, bn_type=nn.BatchNorm2d):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)

        #self.bn1 = nn.BatchNorm2d(mip)
        self.bn1 = bn_type(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)

        return a_w * a_h


class MixAttention(nn.Module):
    def __init__(self, dim):
        super(MixAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        return pattn2


class MixAttentionSelectFusion(nn.Module):
    def __init__(self, channels, reduction=8, bn_type=nn.BatchNorm2d, eps=1e-8):
        super(MixAttentionSelectFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = CoordAtt(channels, reduction, bn_type)
        self.pa = MixAttention(channels)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)
        self.conv_in = nn.Conv2d(channels, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        # self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.sigma1 = ElementScale(channels, init_value=1e-5, requires_grad=True)
        self.sigma2 = ElementScale(channels, init_value=1e-5, requires_grad=True)
        # self.eps = eps
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        initial = x + y
        initial_in = self.conv_in(initial)
        cattn = self.ca(initial_in)
        sattn = self.sa(initial_in)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        # weights = nn.ReLU()(self.weights)
        # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        result = initial + self.relu(self.sigma1(pattn2 * x)  + self.sigma2(pattn2 * y) )
        result = self.conv(result)
        return result


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )
    def forward(self, x):
        return x * self.scale



class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output



class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        self.dw_conv =  nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1,
                       padding=3//2, dilation=1, groups=dim)
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim//2, dim//2, kernel_size=5, stride=1,
                       padding=5//2, dilation=1, groups=dim//2),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # PW
            nn.Conv2d(dim//2, dim//2, kernel_size=7, stride=1,
                       padding=7//2, dilation=1, groups=dim//2),
            nn.GELU()
        )
        self.conv_init_3 = nn.Sequential(  # PW
            nn.Conv2d(dim//2, dim//2, kernel_size=9, stride=1,
                       padding=9//2, dilation=1, groups=dim//2),
            nn.GELU()
        )
        self.conv_init_4 = nn.Sequential(  # PW
            nn.Conv2d(dim//2, dim//2, kernel_size=1, stride=1,
                       padding=11//2, dilation=1, groups=dim//2),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)
        # self.bn = nn.BatchNorm2d(dim)
        # self.relu = torch.nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()
        self.conv_mix = nn.Conv2d(dim*2, dim*2, 1)
        self.conv_in = nn.Conv2d(dim, dim*2, 1)
        self.conv_out = nn.Conv2d(dim*2, dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv_in(x)
        x = self.dw_conv(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.conv_init_1(x1)
        x2 = self.conv_init_1(x2)
        x3 = self.conv_init_1(x3)
        x4 = self.conv_init_1(x4)
        x0 = torch.cat([x1, x2, x3, x4], dim=1)
        x0 = self.conv_mix(x0)
        x = self.FFC(x0) + x0
        x = self.conv_out(x)
        # x = self.gelu(self.bn(x))
        # x = self.bn(x)
        return x




class ChannelAggregationFFN(nn.Module):

    def __init__(self,
                 channels,
                 feedforward_channels,
                 kernel_size=3,
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = channels
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=channels,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=channels,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.GELU()
        self.dwconv1 = nn.Conv2d(channels, channels//2, 1,1)
        self.dwconv2 = nn.Conv2d(channels, channels//4, kernel_size = 3, stride = 1, padding = 1)
        self.dwconv3 = nn.Conv2d(channels, channels//4, kernel_size = 7, stride = 1, padding = 3)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)
        x = torch.cat([x1,x2,x3],dim=1)        
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class AgentAttention(nn.Module):
    def __init__(self, channels, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert channels % num_heads == 0, f"dim {channels} should be divided by num_heads {num_heads}."
        self.dim = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        self.q1 = nn.Linear(channels, channels, bias=qkv_bias)
        self.q2 = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(channels)
        self.agent_num = agent_num
        # self.dwc = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=1, groups=channels)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)
        self.with_bias_ln1 = WithBias_LayerNorm(channels)
        self.with_bias_ln2 = WithBias_LayerNorm(channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, y):
        b1, c1, h1, w1 = x.shape
        H = h1
        W = w1
        x = x.reshape(b1, c1, -1).transpose(-1, -2).contiguous()
        y = y.reshape(b1, c1, -1).transpose(-1, -2).contiguous()

        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        x = self.with_bias_ln1(x)
        y = self.with_bias_ln2(y)

        q = self.q1(x)
        q_agent = self.q2(y)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W).contiguous()
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3).contiguous()
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3).contiguous()
        k, v = kv[0], kv[1]
        agent_tokens = self.pool(q_agent.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1).contiguous()
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1))
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1))
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2).contiguous()
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        # x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-1, -2).reshape(b1, c1, h1, w1).contiguous()

        return x




class MainBlock(nn.Module):
    def __init__(self, channels):
        super(MainBlock, self).__init__()

        self.ffcm = Freq_Fusion(channels)
        self.conv_in = nn.Conv2d(channels, channels, 1)
        # self.norm = MakeLayerNorm(channels)
        self.conv2_1 = nn.Conv2d(channels, channels, (1, 21), padding=(0, 10), groups=channels)
        self.conv2_2 = nn.Conv2d(channels, channels, (21, 1), padding=(10, 0), groups=channels)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels=channels*4, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        shortcut = x
        # x = self.norm(x)
        x = self.conv_in(x)
        strip1 = self.conv2_1(x)
        strip2 = self.conv2_2(x)
        ffcm = self.ffcm(x)
        x = torch.cat([shortcut,  strip1, ffcm, strip2], dim=1)
        x = self.conv_fusion(x) 
        return x

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch, out_ch, kernel_size=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.conv_bn_relu_high = nn.Sequential(
                                    nn.Conv2d(in_ch*3, out_ch, kernel_size=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        # x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        l = self.conv_bn_relu(yL)
        h = torch.cat([y_HL, y_LH, y_HH], dim=1)
        h = self.conv_bn_relu_high(h)
        return l, h
class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class HLAEM(nn.Module): #高低频特征自适应
    def __init__(self, channels):
        super(HLAEM, self).__init__()
        self.conv_block = nn.Sequential(
            # deep wise
            nn.Conv2d(channels, channels*4, kernel_size=1),
            nn.Conv2d(channels*4, channels*4, kernel_size=5, groups=channels, padding=(5 // 2, 5 // 2)),
            nn.GELU(),
            nn.BatchNorm2d(channels*4),
            nn.Conv2d(channels*4, channels, kernel_size=1),
        )
        self.meem = MEEM(channels, channels)
        self.alise1 = nn.Conv2d(2 * channels, channels, 1, 1, 0)  # one_module(n_feats)
        self.alise2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            )  
        self.att = CALayer(channels)
        self.down = Down_wt(3, channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        low, high = self.down(x)
        low_feat = self.meem(low)
        high_feat =  self.conv_block(high)
        out = self.alise2(self.att(self.alise1(torch.cat([low_feat, high_feat], dim=1)))) + low + high
        return out






class AgentAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, bn_type=nn.BatchNorm2d):
        super().__init__()
        self.agent_attn1 = AgentAttention(channels, num_heads=4)
        self.mlp1 = ChannelAggregationFFN(channels, channels*4)
        self.bn1 = bn_type(channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x1, x2):
        attn1 = x1 + self.agent_attn1(x1, x2)
        attn1 = attn1 + self.mlp1(self.bn1(attn1))
        return attn1 

class Model(nn.Module):
    def __init__(self, n_class=6, pretrained = True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        config=[96, 192, 384, 768] # channles of convnext-small
        self.backbone = convnext_small(pretrained,True)
       
        self.MASF1 = MixAttentionSelectFusion(config[1], bn_type=nn.BatchNorm2d)
        self.MASF64 = MixAttentionSelectFusion(config[1], bn_type=nn.BatchNorm2d)

        self.mainblock1 = MainBlock(config[1])
        self.mainblock2 = MainBlock(config[1])

        self.agent_attn1 = AgentAttentionBlock(config[1], num_heads=4)
        self.agent_attn2 = AgentAttentionBlock(config[1], num_heads=4)

        # self.wt_down1 = Down_wt(config[0], config[1])
        # self.steam = HLAEM(config[0]//2)

        self.seg = nn.Sequential(
            nn.Conv2d(config[0]//2, config[0]//2, 1),
            nn.Conv2d(config[0]//2, config[0]//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(config[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config[0]//2, n_class, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.up16_to_64 = nn.Sequential(
            nn.Conv2d(config[3], config[1], 1),
            nn.BatchNorm2d(config[1]),            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )
        self.up32_to_64 = nn.Sequential(

            nn.Conv2d(config[2], config[1], 1),
            nn.BatchNorm2d(config[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.up64_to_128 = nn.Sequential(
            nn.Conv2d(config[1], config[0]//2, 1),
            nn.BatchNorm2d(config[0]//2),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )
        self.down128_to_64 = nn.Sequential(
            nn.BatchNorm2d(config[0]),
            nn.Conv2d(config[0], config[1],kernel_size=2, stride=2)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(config[1]*4, config[1]*2, 1),
            nn.Conv2d(config[1]*2, config[1]*2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(config[1]*2),
            nn.GELU(),
            nn.Conv2d(config[1]*2, config[1], 1),
            nn.Conv2d(config[1], config[1], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(config[1]),
            nn.GELU(),
        )

        self.detail_enhance = HLAEM(config[0]//2)

        # self.attn_block64 = AgentAttentionBlock(config[1], num_heads=4)
        self.attn_block1 = AgentAttentionBlock(config[1], num_heads=4)
        self.attn_block2 = AgentAttentionBlock(config[1], num_heads=4)
        self.MASF_Final = MixAttentionSelectFusion(config[0]//2, bn_type=nn.BatchNorm2d)
    def forward(self, x):
        img = x
        x128,x64,x32,x16 = self.backbone(x)

        x16_to_64 = self.up16_to_64(x16)
        x32_to_64 = self.up32_to_64(x32)

        l64 = self.MASF1(x16_to_64, x32_to_64)
        h64 = self.down128_to_64(x128)
        h64 = self.MASF64(h64, x64)


        l64_att = self.attn_block1(l64, h64)
        h64_att = self.attn_block2(h64, l64)

        l_mian = self.mainblock1(l64_att)
        h_mian = self.mainblock2(h64_att)

        attn_l = self.agent_attn1(l_mian, h_mian) 
        attn_h = self.agent_attn2(h_mian, l_mian) 

        out64 = self.fuse(torch.cat([attn_l, attn_h, h64, l64], dim=1))
        out = self.up64_to_128(out64)

        detal = self.detail_enhance(img)
        out = self.MASF_Final(out, detal)
        out = self.seg(out)
        return out
    
if __name__ == "__main__":
    model = Model(6,False)
    img = torch.rand((1,3,512,512))
    output = model(img)
    print(output.shape)
    
    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total()/1e9))
        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 


        