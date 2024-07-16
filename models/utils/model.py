import logging

import numpy as np

logger = logging.getLogger(__name__)
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
import math
from torchvision.models._utils import IntermediateLayerGetter
import os
from torch.functional import Tensor
from typing import Dict, List
from ..tresnet import TResnetM, TResnetL, TResnetXL
from ..resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from ..swin_transformer import build_swin_transformer
from ..position_encoding import build_position_encoding
from ..misc import clean_state_dict, clean_body_state_dict

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args=None):
        super().__init__(backbone, position_embedding)
        # self.args = args
        if args is not None and 'interpotaion' in vars(args) and args.interpotaion:
            self.interpotaion = True
        else:
            self.interpotaion = False


    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        else:
            # for swin Transformer
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

class WIC_Cls(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(WIC_Cls, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

def S_generation(h, w):
    x = torch.zeros(h * w)
    y = torch.zeros(h * w)

    for i in range(h * w):
        x[i] = i % w
        y[i] = i % h

    x_dist = x.unsqueeze(1) - x.unsqueeze(0)
    y_dist = y.unsqueeze(1) - y.unsqueeze(0)

    dist = torch.sqrt(x_dist ** 2 + y_dist ** 2)
    dist = dist / torch.max(dist)

    for i in range(dist.shape[0]):
        dist[i, i] = 1000000
    return dist

def VDSM(map, loc_dist, min_superpixels):
    c, h, w = map.shape
    map = map.reshape((c, -1)) # (c, hw)

    min_v = torch.min(map, dim=1)
    max_v = torch.max(map, dim=1)
    norm_map = (map - min_v.values.unsqueeze(1)) / (max_v.values.unsqueeze(1) - min_v.values.unsqueeze(1) + 1e-8)

    f = norm_map.transpose(0, 1)

    a = torch.sum(f.unsqueeze(1) * f.unsqueeze(0), dim=2) + (1.0 - loc_dist)
    s = torch.zeros_like(a)

    for i in range(a.shape[0]):
        a_ = a[i, :]
        _, indices = torch.topk(a_, k=2)
        if indices[0] > indices[1]:
            s[indices[1], indices[0]] += 1 / (h * w * loc_dist[indices[1], indices[0]])
        else:
            s[indices[0], indices[1]] += 1 / (h * w * loc_dist[indices[0], indices[1]])

    cls = torch.ones(h * w) * -1
    values, indices = torch.topk(s.reshape(-1), k= h * w - min_superpixels)
    for i in range(values.shape[0]):
        if i > h * w - min_superpixels:
            break
        else:
            if values[i] > 0:
                max_superpixels = torch.max(cls)
                m = indices[i] // (h * w)
                n = indices[i] % (h * w)
                if cls[m] == -1 and cls[n] == -1:
                    cls[m] = int(max_superpixels + 1)
                    cls[m] = int(max_superpixels + 1)
                elif cls[m] == -1 and cls[n] > -1:
                    cls[m] = cls[n]
                elif cls[m] > -1 and cls[n] == -1:
                    cls[n] = cls[m]
                else:
                    cls[cls == cls[n]] = cls[m]
            else:
                pass

    for i in range(h * w):
        if cls[i] == -1:
            max_superpixels = torch.max(cls)
            cls[i] = int(max_superpixels + 1)

    max_superpixels = torch.max(cls)
    clus_feat = []

    map = map.transpose(0, 1) # (hw, c)
    for i in range(int(max_superpixels + 1)):
        mask = torch.zeros_like(cls).cuda()
        mask[cls == i] = 1
        if torch.sum(mask) >= 1:
            feat = torch.sum(mask.unsqueeze(1) * map, dim=0) / torch.sum(mask)
            clus_feat.append(feat.unsqueeze(0))

    return torch.concat(clus_feat, dim=0)


class VoLUNet(nn.Module):

    def __init__(self, backbone, mid_feat, pool, num_classes, subsampling, h, w, min_superpixels):
        super(VoLUNet, self).__init__()
        self.backbone = backbone
        self.pooling = pool
        if subsampling:
            self.subsampling = nn.AdaptiveAvgPool2d((h, w))
            self.min_superpixels = min_superpixels
            self.loc_dist = S_generation(h, w).cuda()
        else:
            self.subsampling = False
            self.min_superpixels = min_superpixels
            self.loc_dist = S_generation(h, w).cuda()
        self.WIC_Cls = WIC_Cls([mid_feat, num_classes])

    def forward(self, x):
        # See note [TorchScript super()]
        feat = self.backbone.forward_features(x)

        x = self.pooling(feat)
        x = torch.flatten(x, 1)

        x = self.WIC_Cls(x)


        if self.training:
            if self.min_superpixels:
                if self.min_superpixels >= feat.shape[2] * feat.shape[3]:
                    b, c, h, w = feat.shape
                    feat_ = feat.transpose(1, 3).reshape((b * h * w, c))
                    feat_ = self.WIC_Cls(feat_)
                    feat_ = feat_.reshape((b * h * w, -1))

                    return x, feat_
                else:
                    if self.subsampling:
                        feat = self.subsampling(feat)
                    b, c, h, w = feat.shape
                    feat_list = []
                    for i in range(b):
                        feat_ = VDSM(feat[i, :, :, :], self.loc_dist, self.min_superpixels)
                        feat_ = self.WIC_Cls(feat_.unsqueeze(0))
                        feat_list.append(feat_.squeeze(0))
                        feats = torch.cat(feat_list, dim=0)
                    return x, feats
            else:
                b, c, h, w = feat.shape
                feat_ = feat.transpose(1, 3).reshape((b * h * w, c))
                feat_ = self.WIC_Cls(feat_)
                feat_ = feat_.reshape((b * h * w, -1))
                return x, feat_

        return x

def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes, 'subsampling':args.subsampling, 'sub_h':args.sub_h, 'sub_w':args.sub_w, 'min_superpixels':args.min_superpixels}
    args = model_params['args']
    args.backbone_name = args.backbone_name.lower()

    if args.backbone_name=='tresnet_m':
        backbone = TResnetM(model_params)
        backbone = load_tresnet(args, backbone)
        model = VoLUNet(backbone, backbone.num_features, backbone.global_pool, model_params['num_classes'], model_params['subsampling'], model_params['sub_h'], model_params['sub_w'], model_params['min_superpixels'])
    elif args.backbone_name=='tresnet_l':
        backbone = TResnetL(model_params)
        backbone = load_tresnet(args, backbone)
        model = VoLUNet(backbone, backbone.num_features, backbone.global_pool, model_params['num_classes'], model_params['subsampling'], model_params['sub_h'], model_params['sub_w'], model_params['min_superpixels'])
    elif args.backbone_name=='resnet18':
        if args.model_path:
            backbone = resnet18(pretrained=True, progress=True, num_classes=model_params['num_classes'])
        else:
            backbone = resnet18(pretrained=False, progress=True, num_classes=model_params['num_classes'])
        model = VoLUNet(backbone, backbone.num_features, backbone.avgpool, model_params['num_classes'], model_params['subsampling'], model_params['sub_h'], model_params['sub_w'], model_params['min_superpixels'])
    elif args.backbone_name=='resnet50':
        if args.model_path:
            backbone = resnet50(pretrained=True, progress=True, num_classes=model_params['num_classes'])
        else:
            backbone = resnet50(pretrained=False, progress=True, num_classes=model_params['num_classes'])
        model = VoLUNet(backbone, backbone.num_features, backbone.avgpool, model_params['num_classes'], model_params['subsampling'], model_params['sub_h'], model_params['sub_w'], model_params['min_superpixels'])
    elif args.backbone_name=='resnet101':
        if args.model_path:
            backbone = resnet101(pretrained=True, progress=True, num_classes=model_params['num_classes'])
        else:
            backbone = resnet101(pretrained=False, progress=True, num_classes=model_params['num_classes'])
        model = VoLUNet(backbone, backbone.num_features, backbone.avgpool, model_params['num_classes'], model_params['subsampling'], model_params['sub_h'], model_params['sub_w'], model_params['min_superpixels'])
    elif args.backbone_name in ['swin_b_224_22k', 'swin_b_384_22k', 'swin_l_224_22k', 'swin_l_384_22k']:

        imgsize = int(args.backbone_name.split('_')[-2])
        backbone = build_swin_transformer(args.backbone_name, imgsize, model_params['num_classes'])
        if args.model_path:
            pretrainedpath = args.model_path
            checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
            from collections import OrderedDict
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head' not in k})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))

        model = VoLUNet(backbone, backbone.num_features, backbone.avgpool, model_params['num_classes'], model_params['subsampling'], model_params['sub_h'], model_params['sub_w'], model_params['min_superpixels'])
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    model

    return model

def load_tresnet(pretrained, model):
    if pretrained.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(pretrained.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
        print("loaded pretrained tresnet!")
    return model

if __name__ == '__main__':
    print(xy_dist(2, 3))