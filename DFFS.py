import sys
sys.path.append('/home/zhensen/medical')
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ResNet import resnet50
#from networks.ResNet import resnet50
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from math import log
from networks.base import BaseNetwork
import math
import torch.utils.model_zoo as model_zoo




__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

resnet_model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x1, x2, x3, x4


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['res2net50_v1b_26w_4s']))
    return model

def relative_pos_dis(height=32, weight=32, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    #dis = torch.exp(-dis*(1/(2*sita**2)))
    return  dis

class CA(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=64, dropout=0., num_patches=1024):
        super(CA, self).__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == in_channels)

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches

        # Project input into query, key, value
        self.qkv_projection = nn.Conv2d(in_channels, inner_dim * 3, kernel_size=3, padding=1, bias=False)

        # Relative position embedding
        self.position_dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).cuda()

        # Head-specific scaling factor
        self.head_scale = nn.Parameter(torch.randn(num_heads), requires_grad=True)

        self.sigmoid = nn.Sigmoid()

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        ) if project_out else nn.Identity()

    def forward(self, input_tensor, mode="train", smooth=1e-4):
        # Project input into q, k, v
        qkv = self.qkv_projection(input_tensor).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.num_heads), qkv)

        # Compute attention map
        attention_map = torch.matmul(q, k.transpose(-1, -2))  # b g n n
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1) + smooth)[:, :, :, None] * torch.sqrt(
            torch.sum(k ** 2, dim=-1) + smooth)[:, :, None, :] + smooth
        attention_map = attention_map / qk_norm

        # Apply position bias
        scale_factor = 1 / (2 * (self.sigmoid(self.head_scale) * (0.4 - 0.003) + 0.003) ** 2)
        position_bias = scale_factor[:, None, None] * self.position_dis[None, :, :]  # g n n
        position_bias = torch.exp(-position_bias)
        position_bias = position_bias / torch.sum(position_bias, dim=-1)[:, :, None]

        if position_bias.shape[1] != attention_map.shape[2]:
            position_bias = F.interpolate(position_bias.unsqueeze(0), size=(attention_map.shape[2], attention_map.shape[3]), mode='bilinear', align_corners=False).squeeze(0)

        # Apply bias to attention map
        attention_map = attention_map * position_bias[None, :, :, :]

        # Compute output from attention and value
        output = torch.matmul(attention_map, v)
        output = rearrange(output, 'b g (h w) d -> b (g d) h w', h=input_tensor.shape[2])

        # Return output
        if mode == "train":
            return self.output_projection(output)
        else:
            return self.output_projection(output), attention_map


class CFF(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class CTR(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=512, dropout=0., num_patches=16):
        super(CTR, self).__init__()

        # Initialize a list of transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CA(dim, num_heads=heads, head_dim=dim_head, dropout=dropout, num_patches=num_patches),
                CFF(dim, mlp_dim, dropout=dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        # Pass input through each layer (attention + feed-forward)
        for attention, feed_forward in self.layers:
            # Apply attention and add the residual connection
            x = attention(x) + x
            # Apply feed-forward and add the residual connection
            x = feed_forward(x) + x
        return x

    def infer(self, x):
        # For inference, track feature tokens and attention maps
        feature_tokens, attention_maps = [], []
        for attention, feed_forward in self.layers:
            # Apply attention in "record" mode and get the output and attention map
            attention_output, attention_map = attention(x, mode="record")
            x = attention_output + x
            # Apply feed-forward and add the residual connection
            x = feed_forward(x) + x
            # Save feature tokens and attention maps for later
            feature_tokens.append(rearrange(x, 'b c h w -> b (h w) c'))
            attention_maps.append(attention_map)

        return x, feature_tokens, attention_maps


class DownSingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.single_conv(x)

class CNNEncoder1(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder1, self).__init__()
        self.scale = 1
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = DownSingleConv(128 // self.scale, out_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        return x

class CNNEncoder2(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder2, self).__init__()
        self.scale = 1
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = DownSingleConv(128 // self.scale, 256 // self.scale)
        self.down3 = DownSingleConv(256 // self.scale, out_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

class CNNEncoder3(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder3, self).__init__()
        self.scale = 1
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = DownSingleConv(128 // self.scale, 256 // self.scale)
        self.down3 = DownSingleConv(256 // self.scale, 512 // self.scale)
        self.down4 = DownSingleConv(512 // self.scale, out_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x

class CNNEncoder4(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder4, self).__init__()
        self.scale = 1
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = DownSingleConv(128 // self.scale, 256 // self.scale)
        self.down3 = DownSingleConv(256 // self.scale, 512 // self.scale)
        self.down4 = DownSingleConv(512 // self.scale, 1024 // self.scale)
        self.down5 = DownSingleConv(1024 // self.scale, out_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return x

class PT(nn.Module):
    def __init__(self, input_channels, input_size):
        super(PT, self).__init__()

        self.input_size = input_size  # Size of the input features (e.g., 80)

        # Define CNN encoders for different patch sizes
        self.encoder1 = CNNEncoder1(256, 64, patch_height=4, patch_width=4)
        self.encoder2 = CNNEncoder2(256, 64, patch_height=8, patch_width=8)
        self.encoder3 = CNNEncoder3(256, 64, patch_height=16, patch_width=16)
        self.encoder4 = CNNEncoder4(256, 64, patch_height=32, patch_width=32)

        # Define Transformer modules for each encoder output
        self.transformer1 = CTR(64, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.1,
                                                 num_patches=16)
        self.transformer2 = CTR(64, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.1,
                                                 num_patches=64)
        self.transformer3 = CTR(64, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.1,
                                                 num_patches=256)
        self.transformer4 = CTR(64, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.1,
                                                 num_patches=1024)

        # Fusion layer to combine features from different transformer outputs
        self.fusion_layer = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, enc_features):
        # Apply CNN encoders to extract features at different patch sizes
        enc1_out = self.encoder1(enc_features)
        enc2_out = self.encoder2(enc_features)
        enc3_out = self.encoder3(enc_features)
        enc4_out = self.encoder4(enc_features)

        # Pass the outputs through the transformer modules
        trans_out1 = self.transformer1(enc1_out)
        trans_out2 = self.transformer2(enc2_out)
        trans_out3 = self.transformer3(enc3_out)
        trans_out4 = self.transformer4(enc4_out)

        # Resize the transformer outputs to match the size of the second transformer output
        trans_out1_resized = F.interpolate(trans_out1, size=trans_out2.size()[2:], mode='bilinear', align_corners=False)
        trans_out3_resized = F.interpolate(trans_out3, size=trans_out2.size()[2:], mode='bilinear', align_corners=False)
        trans_out4_resized = F.interpolate(trans_out4, size=trans_out2.size()[2:], mode='bilinear', align_corners=False)

        # Fuse the transformer outputs by concatenating along the channel dimension
        fused_output = torch.cat([trans_out1_resized, trans_out2, trans_out3_resized, trans_out4_resized], dim=1)

        # Apply the fusion layer (optional step for channel reduction)
        fused_output = self.fusion_layer(fused_output)

        return fused_output



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DualAttentionModule(nn.Module):
    def __init__(self):
        super(DualAttentionModule, self).__init__()

        # Initial prediction block with convolutional layers for feature processing
        self.initial_pred_block = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # Refinement layers for feature extraction and interaction
        self.refine_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.refine_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.refine_conv3 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, input_features, attention_map):
        # Upsample the attention map to match the spatial size of input features
        upsampled_attention = F.interpolate(attention_map, input_features.size()[2:], mode='bilinear', align_corners=False)

        # Initial prediction by concatenating input features with upsampled attention map
        initial_features = self.initial_pred_block(torch.cat([input_features, upsampled_attention], dim=1))

        # Create an inverse attention mask and apply it to the input features
        inverted_attention = -1 * (torch.sigmoid(upsampled_attention)) + 1
        inverted_attention = inverted_attention.expand(-1, 64, -1, -1)
        refined_features = inverted_attention * input_features

        # Apply refinement convolutions
        refined_features = F.relu(self.refine_conv1(refined_features))
        refined_features = F.relu(self.refine_conv2(refined_features))

        # Final refinement step
        final_refined_features = self.refine_conv3(refined_features)

        # Combine the initial features, upsampled attention map, and refined features
        output_features = final_refined_features + upsampled_attention + initial_features

        return output_features


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x




class ImprovedT2CModule(nn.Module):
    def __init__(self, cnn_channels, vit_channels, reduction=16):
        super(ImprovedT2CModule, self).__init__()

        # Channel Attention Components
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pwconv1 = nn.Conv2d(vit_channels, vit_channels // reduction, kernel_size=1)
        self.pwconv2 = nn.Conv2d(vit_channels // reduction, vit_channels, kernel_size=1)

        # Spatial Attention Components
        self.dsc_5x5 = DepthwiseSeparableConv(vit_channels, kernel_size=5)
        self.dsc_7x7 = DepthwiseSeparableConv(vit_channels, kernel_size=7)

        # Channel Mapper to align dimensions
        self.channel_mapper_spatial = nn.Conv2d(vit_channels, cnn_channels, kernel_size=1, bias=False)
        self.channel_mapper_channel = nn.Conv2d(vit_channels, cnn_channels, kernel_size=1,
                                                bias=False)  # New mapping layer

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, cnn_features, vit_features):
        # --------- Channel Attention ---------
        channel_weights = self.global_avg_pool(vit_features)
        channel_weights = F.relu(self.pwconv1(channel_weights))
        channel_weights = self.sigmoid(self.pwconv2(channel_weights))
        channel_attention = vit_features * channel_weights  # Channel-wise recalibration

        # Map channel_attention to cnn_channels
        channel_attention_mapped = self.channel_mapper_channel(channel_attention)

        # --------- Spatial Attention ---------
        dsc_5x5_out = self.dsc_5x5(vit_features)
        dsc_7x7_out = self.dsc_7x7(vit_features)
        spatial_attention = dsc_5x5_out + dsc_7x7_out
        spatial_attention = self.sigmoid(self.channel_mapper_spatial(spatial_attention))

        # --------- Combine Channel and Spatial Attention with Addition ---------
        attention_weights = channel_attention_mapped + spatial_attention  # Additive fusion
        attention_weights = self.sigmoid(attention_weights)

        # --------- Refine CNN Features ---------
        refined_features = cnn_features * attention_weights

        return refined_features


class DCHN(nn.Module):
    def __init__(self, num_classes):
        super(DCHN, self).__init__()

        # Initialize ResNet backbone with custom Res2Net architecture
        self.backbone = res2net50_v1b_26w_4s(pretrained=True)

        # Multi-scale transformer module to capture richer features
        self.multi_scale_transformer = PT(256, 64)

        # Total number of segmentation classes
        self.num_classes = num_classes
        self.T2C = ImprovedT2CModule(2048,256)

        # 1x1 convolutions to reduce channels at various stages
        self.reduce_x1 = Conv1x1(256, 64)
        self.reduce_x2 = Conv1x1(512, 64)
        self.reduce_x3 = Conv1x1(1024, 64)
        self.reduce_x4 = Conv1x1(2048, 64)


        # Dual Attention Mechanism modules for feature refinement
        self.attention1 = DualAttentionModule()
        self.attention2 = DualAttentionModule()
        self.attention3 = DualAttentionModule()
        self.attention4 = DualAttentionModule()

        # Final segmentation prediction layers at different scales
        self.segmentation_head1 = nn.Conv2d(64, self.num_classes, kernel_size=1)
        self.segmentation_head2 = nn.Conv2d(64, self.num_classes, kernel_size=1)
        self.segmentation_head3 = nn.Conv2d(64, self.num_classes, kernel_size=1)
        self.segmentation_head4 = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def forward(self, input_tensor):
        # Extract feature maps from ResNet backbone
        low_res_features, mid_res_features, high_res_features, deepest_features = self.backbone(input_tensor)

        # Pass the first feature map through the multi-scale transformer
        transformed_features = self.multi_scale_transformer(low_res_features)

        # Upsample the transformed features to match the size of the deepest feature map
        transformed_features = F.interpolate(transformed_features, size=deepest_features.shape[2:], mode='bilinear', align_corners=False)
        # Apply dual attention mechanism for feature fusion
        refined_attention = self.T2C(deepest_features,transformed_features)
        # Apply 1x1 convolutions to reduce the number of channels
        reduced_low_res = self.reduce_x1(low_res_features)
        reduced_mid_res = self.reduce_x2(mid_res_features)
        reduced_high_res = self.reduce_x3(high_res_features)
        reduced_deepest_res = self.reduce_x4(deepest_features)

        # Further reduce the refined attention feature map
        refined_attention = self.reduce_x4(refined_attention)

        # Apply dual attention to refine features across different resolutions
        level4_output = self.attention4(reduced_deepest_res, refined_attention)
        level3_output = self.attention3(reduced_high_res, level4_output)
        level2_output = self.attention2(reduced_mid_res, level3_output)
        level1_output = self.attention1(reduced_low_res, level2_output)

        output_at_level1 = self.segmentation_head1(level1_output)
        output_at_level1 = F.interpolate(output_at_level1, scale_factor=4, mode='bilinear', align_corners=False)

        # Return the final segmentation output (most refined resolution)
        return output_at_level1
