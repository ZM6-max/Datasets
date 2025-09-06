import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/data/tangchen/deeplabv3-plus-pytorch-main')
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from torch import nn, Tensor
from timm.models.layers import DropPath, trunc_normal_
import math

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
    
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class CatKey(nn.Module):
    def __init__(self, pool_ratio=[1,2,4,8], dim=[256,160,64,32]):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.ModuleList([nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])
        self.pool_list = nn.ModuleList([nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])

    def forward(self, x):
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                out_list.append(x[i])
        return torch.cat(out_list, dim=1)

class CatKeyMulti(nn.Module):
    def __init__(self, pool_ratio=[1,2,4,8], dim=[256,160,64,32], num_feat = 4):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.ModuleList([nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])
        self.pool_list = nn.ModuleList([nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])
        for _ in range(num_feat):
            self.sr_list.append(nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1))
            self.pool_list.append(nn.AvgPool2d(self.pool_ratio[1], self.pool_ratio[1], ceil_mode=True))

    def forward(self, x, xMulti):
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                out_list.append(x[i])
        for l in range(len(xMulti)): #for the middle feature at stage 3
            out_list.append(self.sr_list[cnt](self.pool_list[cnt](xMulti[l])))
            cnt += 1
        return torch.cat(out_list, dim=1)

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            # x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
            # x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
            x_ = self.norm(y)
            x_ = self.act(y)
        else:
            x_ = y
            
        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4) #여기에다가 rollout을 넣는다면?
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2, H1, W1):
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2)) #self.norm2(y)
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))

        # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)
        # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x
    
class APFormerHead(nn.Module):
    """
    Attention-Pooling Former
    """
    def __init__(self, pool_scales=(1, 2, 3, 6), in_channels = [64, 128 ,320, 512],embedding_dim = 128,num_classes=17,num_heads=[8, 5, 2, 1],pool_ratio=[1, 2, 4, 8]):
        super(APFormerHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)
        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=2,
                                drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=2,
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=2,
                                drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=2,
                                drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
        )
        self.Dy32 = DySample(c4_in_channels, 8, 'pl', groups=8)
        self.Dy16 = DySample(c3_in_channels, 4, 'pl', groups=8)
        self.Dy8 = DySample(c2_in_channels, 2, 'pl')
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c4 = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c3 = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c2 = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c1 = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1, c_key, h4, w4, h1, w1)
        _c4 = self.Dy32(_c4)
        _c3 = self.Dy16(_c3) 
        _c2 = self.Dy8(_c2)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)
        # print(_c1.shape)
        # print(_c2.shape)
        # print(_c3.shape)
        # print(_c4.shape)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    
def unit_test():
    num_minibatch = 1
    x1 = torch.randn(4, 64, 128, 128).cuda(0)
    x2 = torch.randn(4, 128, 64, 64).cuda(0)
    x3 = torch.randn(4, 320, 32, 32).cuda(0)
    x4 = torch.randn(4, 512, 16, 16).cuda(0)
    model   = APFormerHead().cuda(0)
    # from thop import profile
    from fvcore.nn import flop_count_table, FlopCountAnalysis
    print(flop_count_table(FlopCountAnalysis(model,[x1,x2,x3,x4])))
    # total_ops, total_params  = profile(rtf_net, inputs=(input ,), verbose=False)
    # print("%s | %s" % ("Params(M)", "FLOPs(G)"))
    # print("%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))


if __name__ == '__main__':
    unit_test()
    